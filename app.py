from flask import Flask, request, jsonify, render_template
import numpy as np

# --- 1. CLASES DE LÓGICA DE SIMULACIÓN (ESTABLE Y CORREGIDA) ---

class SimulationParams:
    """Almacena los parámetros físicos y computacionales para la simulación MD."""
    def __init__(self, N, NS, dt, T_0, ign, sig=1.0, eps=1.0, r_ctf=2.5, bs=4.0):
        self.N = N
        self.NS = NS
        self.dt = dt
        self.T_0 = T_0
        self.ign = ign
        self.sig = sig
        self.eps = eps
        self.r_ctf = r_ctf
        self.bs = bs
        
        self.L = self.bs * self.sig
        self.Volume = self.L ** 3
        
        self.rcut2 = self.r_ctf ** 2
        self.sig6 = self.sig ** 6
        self.sig12 = self.sig ** 12
        self.eps4 = 4.0 * self.eps

class MolecularDynamicsSimulatorPython:
    """Clase para la lógica de MD, ahora diseñada para ser controlada por un servidor."""
    def __init__(self):
        self.is_initialized = False
        self.current_step = 0
        self.params = None
        self.particles = {}
        self.simulation_data = {'T': [], 'E': [], 'P': []}
        self.instant_values = {}

    def _apply_pbc(self, r_vec):
        """Aplica Condiciones de Contorno Periódicas (PBC) usando la Convención de Imagen Mínima (MIC)."""
        L = self.params.L
        dr = r_vec / L
        # Aplicar el desplazamiento periódico
        r_vec = r_vec - L * np.round(dr)
        return r_vec

    def _calculate_forces_and_energy(self):
        """Calcula fuerzas de LJ y Energía Potencial Total (U_total) y Virial."""
        p = self.params
        N = p.N
        
        self.particles['a_old'] = self.particles['a'].copy()
        self.particles['a'] = np.zeros((N, 3))
        U_total = 0.0
        virial = 0.0 

        for i in range(N):
            for j in range(i + 1, N):
                r_vec = self.particles['r'][i, :] - self.particles['r'][j, :]
                r_vec_pbc = self._apply_pbc(r_vec) # MIC
                r2 = np.sum(r_vec_pbc ** 2)

                if r2 < 1e-6: 
                     continue 

                if r2 < p.rcut2:
                    inv_r2 = 1.0 / r2
                    inv_r6 = inv_r2 * inv_r2 * inv_r2
                    
                    U_ij = p.eps4 * (p.sig12 * inv_r6 * inv_r6 - p.sig6 * inv_r6)
                    U_total += U_ij

                    force_magnitude_factor = p.eps4 * inv_r2 * (
                        12.0 * p.sig12 * inv_r6 * inv_r6 - 6.0 * p.sig6 * inv_r6
                    )
                    
                    F_ij = force_magnitude_factor * r_vec_pbc
                    
                    self.particles['a'][i, :] += F_ij
                    self.particles['a'][j, :] -= F_ij
                    
                    virial += np.dot(F_ij, r_vec_pbc)
        
        return U_total, virial

    def _update_positions_velocities(self):
        """Paso de integración de Velocity Verlet."""
        dt = self.params.dt
        dt2_2 = dt * dt / 2.0
        dt_2 = dt / 2.0
        L = self.params.L

        # 1. Actualizar posiciones (r(t+dt))
        self.particles['r'] += self.particles['v'] * dt + self.particles['a_old'] * dt2_2
        # Aplicar PBC a las posiciones: asegura que r esté en el rango [0, L]
        self.particles['r'] = np.mod(self.particles['r'], L) 

        # 2. Recalcular fuerzas (a(t+dt))
        KE_old = 0.5 * np.sum(self.particles['v'] ** 2)
        U_total_new, virial_new = self._calculate_forces_and_energy() # Esto actualiza self.particles['a']

        # 3. Actualizar velocidades (v(t+dt))
        self.particles['v'] += (self.particles['a_old'] + self.particles['a']) * dt_2
        
        KE = 0.5 * np.sum(self.particles['v'] ** 2)

        return KE, U_total_new, virial_new

    def _calculate_instantaneous_values(self, KE, PE, virial):
        """Calcula y almacena los valores termodinámicos instantáneos."""
        N = self.params.N
        Volume = self.params.Volume
        
        temp = (2.0 * KE) / (3.0 * N)
        
        # Presión corregida: P = (N*T/V) - (Virial / (3V))
        pressure = (N * temp / Volume) - (virial / (3.0 * Volume))
        
        Z = (pressure * Volume) / (N * temp) if (N * temp > 1e-9) else 0.0
        
        E_total = KE + PE
        
        self.instant_values = {
            'T': temp, 'P': pressure, 'E_total': E_total, 'Z': Z,
        }

    def initialize(self, params_dict):
        """Inicializa el sistema MD con nuevos parámetros."""
        self.params = SimulationParams(**params_dict)
        self.current_step = 0
        self.simulation_data = {'T': [], 'E': [], 'P': []}
        
        N = self.params.N
        L = self.params.L
        
        # Inicialización de posiciones (retículo SC)
        n_side = int(np.ceil(N ** (1/3)))
        spacing = L / n_side
        p_count = 0
        r_list = []
        v_list = []
        
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    if p_count < N:
                        # Posiciones centradas
                        r_list.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
                        v_list.append(np.random.randn(3)) # Usar randn para distribución normal
                        p_count += 1
        
        self.particles['r'] = np.array(r_list)
        
        # Añadir Jitter para estabilidad (romper la red perfecta)
        max_jitter = 0.05 * spacing # 5% del espaciado
        self.particles['r'] += np.random.uniform(-max_jitter, max_jitter, self.particles['r'].shape)

        self.particles['r'] = np.mod(self.particles['r'], L)

        self.particles['v'] = np.array(v_list)
        self.particles['a'] = np.zeros((N, 3))
        self.particles['a_old'] = np.zeros((N, 3))
        
        # Escalado de Velocidad para T_0 
        initial_KE_unscaled = 0.5 * np.sum(self.particles['v'] ** 2)
        initial_T_unscaled = (2 * initial_KE_unscaled) / (3 * N)
        scale_factor = np.sqrt(self.params.T_0 / initial_T_unscaled)
        self.particles['v'] *= scale_factor
        
        # Centrado del momento (Momentum=0)
        current_momentum = np.sum(self.particles['v'], axis=0)
        self.particles['v'] -= current_momentum / N
        
        initial_KE_scaled = 0.5 * np.sum(self.particles['v'] ** 2)
        
        U_initial, virial_initial = self._calculate_forces_and_energy() 
        self._calculate_instantaneous_values(initial_KE_scaled, U_initial, virial_initial)

        self.is_initialized = True
        
        return {
            "L": L, 
            "N": N, 
            "r": self.particles['r'][:, :2].tolist(), # Solo X, Y para la UI
            "T": self.instant_values['T'],
            "E": self.instant_values['E_total'],
            "P": self.instant_values['P'],
            "Z": self.instant_values['Z'] 
        }

    def do_step(self):
        # ... (El resto de la lógica do_step es la misma y está correcta) ...
        if not self.is_initialized:
            return {"error": "El simulador no está inicializado."}

        self.current_step += 1
        
        KE, PE, virial = self._update_positions_velocities()
        self._calculate_instantaneous_values(KE, PE, virial)
        
        if self.current_step >= self.params.ign:
            self.simulation_data['T'].append(self.instant_values['T'])
            self.simulation_data['E'].append(self.instant_values['E_total'])
            self.simulation_data['P'].append(self.instant_values['P'])
            
        return {
            "step": self.current_step,
            "r": self.particles['r'][:, :2].tolist(), 
            "T": self.instant_values['T'],
            "E": self.instant_values['E_total'],
            "P": self.instant_values['P'],
            "Z": self.instant_values['Z'], 
            "is_equilibrium": self.current_step >= self.params.ign
        }
    
    def calculate_averages(self):
        # ... (La lógica calculate_averages es la misma y está correcta) ...
        if not self.simulation_data['E']:
            return 0.0, 0.0, 0.0 

        avg_T = np.mean(self.simulation_data['T'])
        avg_E = np.mean(self.simulation_data['E'])
        avg_P = np.mean(self.simulation_data['P'])
        
        return avg_T, avg_E, avg_P


# --- 2. SERVIDOR FLASK (CONFIGURACIÓN CORREGIDA) ---

# ********** CORRECCIÓN CRÍTICA **********
# Vuelve a la configuración estándar de Flask para que encuentre 'simulador.html'
app = Flask(__name__, template_folder='templates', static_folder='static') 
simulator_instance = MolecularDynamicsSimulatorPython()

@app.route('/')
def serve_html():
    # Flask buscará 'simulador.html' dentro de la carpeta 'templates'
    return render_template('simulador.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_simulation():
    data = request.json
    try:
        data['N'] = int(data.get('N', 32)) 
        data['NS'] = int(data.get('NS', 5000))
        data['dt'] = float(data.get('dt', 0.005))
        data['T_0'] = float(data.get('T_0', 1.0))
        data['ign'] = int(data.get('ign', 200)) 
        data['bs'] = float(data.get('bs', 4.0))

        initial_state = simulator_instance.initialize(data)
        
        return jsonify(initial_state), 200
    except Exception as e:
        print(f"Error grave en la inicialización: {e}")
        return jsonify({"error": f"Error grave en la inicialización (ver consola de Python): {str(e)}"}), 400

@app.route('/api/step', methods=['POST'])
def run_step():
    try:
        result = simulator_instance.do_step()
        if "error" in result:
             return jsonify(result), 400
        return jsonify(result), 200
    except Exception as e:
        print(f"Error durante el paso de MD: {e}")
        return jsonify({"error": f"Error durante el paso de MD (ver consola de Python): {str(e)}"}), 500

@app.route('/api/averages', methods=['GET'])
def get_averages():
    try:
        avg_T, avg_E, avg_P = simulator_instance.calculate_averages()
        return jsonify({"avg_T": avg_T, "avg_E": avg_E, "avg_P": avg_P})
    except Exception as e:
        return jsonify({"error": "No hay datos para promediar."}), 400


if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    print("Abre tu navegador en: http://127.0.0.1:5000/")
    # Quitar debug=True para producción
    app.run(debug=True)