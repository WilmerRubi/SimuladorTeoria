from flask import Flask, request, jsonify, render_template
from flask import send_file
import numpy as np
import io
import csv

# --- 1. CLASES DE LÓGICA DE SIMULACIÓN ---

class SimulationParams:
        """Almacena los parámetros físicos y computacionales para la simulación MD.

        Comentarios generales:
        - Esta clase centraliza las constantes y parámetros que definen la
            simulación: número de partículas `N`, número de pasos `NS`, paso de
            tiempo `dt`, temperatura objetivo `T_0`, parámetros del potencial
            Lennard-Jones (`sig`, `eps`, `r_ctf`), tamaño de caja `bs`, masa por
            defecto y tipo de pared.
        - También guarda flags para activar características opcionales como
            gravedad (`use_gravity`) y colisiones rígidas (`hard_sphere`), junto
            con parámetros asociados (constante gravitacional `G`, radios,
            coeficientes de restitución).
        - `L` y `Volume` derivan del tamaño de caja `bs` y se usan para aplicar
            condiciones de frontera y cómputos de densidad/ presión.
        - Notas: los valores por defecto están en unidades reducidas (σ = 1).
        """
    def __init__(self, N, NS, dt, T_0, ign, sig=1.0, eps=1.0, r_ctf=2.5, bs=4.0, mass=1.0, wall_type='periodic', restitution=1.0,
                 use_gravity=False, G=1.0, hard_sphere=False, particle_radius=None, collision_restitution=1.0):
        self.N = N
        self.NS = NS
        self.dt = dt
        self.T_0 = T_0
        self.ign = ign
        self.sig = sig
        self.eps = eps
        self.r_ctf = r_ctf
        self.bs = bs
        self.mass = mass
        self.wall_type = wall_type 
        self.restitution = restitution

        self.use_gravity = use_gravity
        self.G = G
        self.hard_sphere = hard_sphere
      
        self.particle_radius = particle_radius
        self.collision_restitution = collision_restitution
        self.L = self.bs * self.sig
        self.Volume = self.L ** 3

        self.rcut2 = self.r_ctf ** 2
        self.sig6 = self.sig ** 6
        self.sig12 = self.sig ** 12
        self.eps4 = 4.0 * self.eps

class MolecularDynamicsSimulatorPython:
        """Clase para la lógica de la simulación Molecular Dynamics (MD).

        Responsabilidades principales:
        - Mantener el estado de todas las partículas (`self.particles`):
            posiciones `r`, velocidades `v`, aceleraciones `a`, masas `m` y radios.
        - Inicializar configuraciones (red FCC o entrada desde CSV/cliente) y
            normalizar velocidades a la temperatura objetivo.
        - Calcular fuerzas entre partículas: potencial Lennard-Jones por defecto
            y, opcionalmente, atracción newtoniana (gravedad) si `use_gravity`.
        - Integrar las ecuaciones de movimiento mediante un esquema tipo Verlet
            y aplicar termostato de Berendsen durante la fase de ignición.
        - Detectar y resolver colisiones rígidas (modo `hard_sphere`) usando
            resolución por impulsos y corrección posicional simple.

        Advertencia de rendimiento: las rutinas de fuerza y colisión usan un
        doble bucle sobre pares (O(N^2)). Para sistemas con muchas partículas
        es recomendable implementar neighbor-lists (cell-lists / Verlet lists).
        """
    def __init__(self):
        self.is_initialized = False
        self.current_step = 0
        self.params = None
        self.particles = {}
        self.instant_values = {'T': 0.0, 'E_total': 0.0, 'P': 0.0, 'Z': 0.0}
        self.average_values = {'T': 0.0, 'E': 0.0, 'P': 0.0, 'count': 0}
        
    def _initialize_fcc(self, N, L):
        """Inicializa las posiciones en una red FCC con N partículas."""
        n_cell = int(np.ceil((N / 4)**(1/3))) 
        r = []
        base_positions = np.array([
            [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]
        ])
        
        a = L / n_cell 
        
        count = 0
        for i in range(n_cell):
            for j in range(n_cell):
                for k in range(n_cell):
                    for base in base_positions:
                        if count < N:
                            pos = (np.array([i, j, k]) + base) * a
                            r.append(pos)
                            count += 1
                        else:
                            break
                    if count >= N:
                        break
                if count >= N:
                    break
            if count >= N:
                break

        return np.array(r, dtype=np.float64)


    def initialize(self, params_dict):
        """Inicializa o reinicializa el estado de la simulación."""
        # --- Qué hace `initialize()` ---
        # 1) Lee parámetros físicos y de ejecución provistos por el cliente.
        # 2) Construye la configuración inicial de posiciones (red FCC por
        #    defecto o entrada explícita desde `particles`) y genera
        #    velocidades aleatorias que se escalan para la temperatura inicial.
        # 3) Inicializa arrays de masas y radios por partícula.
        # 4) Calcula fuerzas iniciales y valores instantáneos (T, E, P, Z).
        N = int(params_dict.get('N', 50))
        NS = int(params_dict.get('NS', 5000))
        dt = float(params_dict.get('dt', 0.005))
        T_0 = float(params_dict.get('T_0', 1.0))
        ign = int(params_dict.get('ign', 200))
        bs = float(params_dict.get('bs', 4.0))
        sig = float(params_dict.get('sig', 1.0))
        eps = float(params_dict.get('eps', 1.0))
        mass = float(params_dict.get('mass', 1.0))
        wall_type = params_dict.get('wall_type', 'periodic')
        restitution = float(params_dict.get('restitution', 1.0))
        use_gravity = bool(params_dict.get('use_gravity', False))
        G = float(params_dict.get('G', 1.0))
        hard_sphere = bool(params_dict.get('hard_sphere', False))
        particle_radius = params_dict.get('particle_radius', None)
        collision_restitution = float(params_dict.get('collision_restitution', restitution))

        self.params = SimulationParams(N=N, NS=NS, dt=dt, T_0=T_0, ign=ign, bs=bs, sig=sig, eps=eps, mass=mass, wall_type=wall_type, restitution=restitution,
                           use_gravity=use_gravity, G=G, hard_sphere=hard_sphere, particle_radius=particle_radius, collision_restitution=collision_restitution)
        L = self.params.L
        
        # 1. Posiciones y velocidades
        particles_input = params_dict.get('particles', None)
        if particles_input:

            r_list = []
            v_list = []
            m_list = []
            for p in particles_input:
                pos = p.get('r', [0.0, 0.0, 0.0])
                vel = p.get('v', [0.0, 0.0, 0.0])
                mass_p = float(p.get('m', mass))
                r_list.append(pos)
                v_list.append(vel)
                m_list.append(mass_p)

            r = np.array(r_list, dtype=np.float64)
            v = np.array(v_list, dtype=np.float64)

            if r.shape[0] != N:
                N = r.shape[0]
                self.params.N = N
        else:
            r = self._initialize_fcc(N, L)
            
            v = np.random.randn(N, 3) 
        v -= np.mean(v, axis=0) 
        
   
        current_T = np.sum(v**2) / (3 * N - 3) 
        scale_factor = np.sqrt(T_0 / current_T)
        v *= scale_factor
        
        # 3. Aceleraciones, masas y Variables de estado
        m_array = np.full(N, self.params.mass, dtype=np.float64)
        if particles_input:
  
            m_array = np.array(m_list, dtype=np.float64)


        if particles_input and any('r' in p and 'm' in p for p in particles_input):
    
            pass

        if particle_radius is None:
            radii = np.full(N, 0.5 * self.params.sig, dtype=np.float64)
        else:
            if isinstance(particle_radius, (list, tuple, np.ndarray)):
                radii = np.array(particle_radius, dtype=np.float64)
                if radii.shape[0] != N:
               
                    radii = np.resize(radii, N)
            else:
                radii = np.full(N, float(particle_radius), dtype=np.float64)

        self.particles = {
            'r': r,
            'v': v,
            'a': np.zeros((N, 3), dtype=np.float64),
            'm': m_array,
            'radius': radii
        }
        
        self.current_step = 0
        self.is_initialized = True
        self.average_values = {'T': 0.0, 'E': 0.0, 'P': 0.0, 'count': 0}

        # Calcular fuerzas y energía iniciales 
        self.compute_forces() 
        self._compute_instantaneous_values() # Calcula T, E, P, Z iniciales
        
        return {
            "L": L, 
            "N": N, 
            "r": self.particles['r'].tolist(),
            "T": self.instant_values['T'],
            "E": self.instant_values['E_total'],
            "P": self.instant_values['P'],
            "Z": self.instant_values['Z'] 
        }

    def _minimum_image_convention(self, r_ij):
        """Aplica la convención de la imagen mínima (Periodic Boundary Conditions)."""
        L = self.params.L
        r_ij[r_ij > L/2] -= L
        r_ij[r_ij < -L/2] += L
        return r_ij

    def compute_forces(self):
        """Calcula fuerzas, energía potencial y presión viral."""
        # --- Flujos principales de `compute_forces()` ---
        # Recorre todos los pares (i < j) y:
        #  - Aplica la convención de la imagen mínima para PBC.
        #  - Si r_ij está dentro del corte (`rcut2`) calcula la fuerza
        #    Lennard-Jones y contribuye a la aceleración de cada partícula.
        #  - Si `use_gravity` está activado se añade la fuerza de atracción
        #    newtoniana entre pares (proporcional a m_i * m_j / r^2).
        #  - Acumula energía potencial y el término virial para la presión.
        #
        # Resultado: actualiza `self.particles['a']` con la aceleración total
        # por partícula y guarda `E_pot` y `virial` en `self.instant_values`.
        r = self.particles['r']
        N = self.params.N
        L = self.params.L
        eps4 = self.params.eps4
        sig6 = self.params.sig6
        sig12 = self.params.sig12
        rcut2 = self.params.rcut2
        m = self.particles.get('m', np.ones(N, dtype=np.float64))

        a = np.zeros((N, 3), dtype=np.float64)
        E_pot = 0.0
        virial = 0.0

        G = self.params.G if hasattr(self.params, 'G') else 0.0

        for i in range(N):
            for j in range(i + 1, N):
                r_ij = r[i] - r[j]
                r_ij = self._minimum_image_convention(r_ij)
                r2 = np.sum(r_ij**2)

                if r2 < rcut2 and r2 > 1e-12:
                    r6 = r2**3
                    r12 = r6**2

                    scalar = eps4 * (12.0 * sig12 / r12 - 6.0 * sig6 / r6) / r2
                    F_ij = scalar * r_ij

               
                    a[i] += F_ij / m[i]
                    a[j] -= F_ij / m[j]

                    E_pot += eps4 * (sig12 / r12 - sig6 / r6)
                    virial += np.dot(F_ij, r_ij)
          
                if self.params.use_gravity and r2 > 1e-12:
             
                    fg_mag = G * m[i] * m[j] / r2
                    n_ij = r_ij / np.sqrt(r2)
                    Fg = -fg_mag * n_ij  
                    a[i] += Fg / m[i]
                    a[j] -= Fg / m[j]

        self.particles['a'] = a
        self.instant_values['E_pot'] = E_pot
        self.instant_values['virial'] = virial

    def _compute_instantaneous_values(self):
        """Calcula T, E, P, Z en el estado actual."""
        # Calcula magnitudes termodinámicas instantáneas a partir del estado:
        # - Energía cinética (teniendo en cuenta masas individuales)
        # - Temperatura (en unidades reducidas, k_B = 1 en estas unidades)
        # - Energía total (cinética + potencial)
        # - Presión mediante la relación que incluye el término virial
        # - Factor Z = P / (rho * T)
        v = self.particles['v']
        N = self.params.N
        Volume = self.params.Volume
        m = self.particles.get('m', np.ones(N, dtype=np.float64))
        # Energía cinética considerando masas: 0.5 * sum(m_i * v_i^2)
        E_kin = 0.5 * np.sum(m * np.sum(v**2, axis=1))

        # Temperatura en unidades reducidas
        self.instant_values['T'] = 2.0 * E_kin / (3.0 * N)

        self.instant_values['E_total'] = E_kin + self.instant_values['E_pot']

        rho = N / Volume
        virial_sum = self.instant_values['virial']
        self.instant_values['P'] = rho * self.instant_values['T'] + (1.0 / (3.0 * Volume)) * virial_sum
        
        self.instant_values['Z'] = self.instant_values['P'] / (rho * self.instant_values['T']) if self.instant_values['T'] > 0 else 0.0

    def _update_averages(self):
        """Actualiza los promedios después del periodo de ignición."""
        if self.current_step > self.params.ign:
            self.average_values['T'] += self.instant_values['T']
            self.average_values['E'] += self.instant_values['E_total']
            self.average_values['P'] += self.instant_values['P']
            self.average_values['count'] += 1

    def _integrate_step(self):
        """Implementa el algoritmo de Verlet de forma simplificada."""
        # --- Esquema de integración (por paso) ---
        # 1) Avanzar posiciones r(t+dt) = r(t) + v*dt + 0.5*a*dt^2
        # 2) Aplicar condiciones de frontera: periódicas (PBC) o reflectivas.
        # 3) Guardar aceleración antigua a(t).
        # 4) Calcular nuevas fuerzas → a(t+dt) llamando a compute_forces().
        # 5) (Si está activado) resolver colisiones rígidas (hard-sphere):
        #    detectar solapamientos, aplicar impulso normal y corrección
        #    posicional para evitar interpenetración.
        # 6) Actualizar velocidades v(t+dt) = v(t) + 0.5*(a(t)+a(t+dt))*dt.
        # 7) Aplicar termostato de Berendsen durante la fase de ignición
        #    (si current_step < ign) para acoplar rápidamente la temperatura.
        r = self.particles['r']
        v = self.particles['v']
        a = self.particles['a']
        dt = self.params.dt
        L = self.params.L

        # 1. Actualizar posiciones (r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2)
        r += v * dt + 0.5 * a * dt**2

        # 1.b Aplicar condiciones de frontera según parámetro
        if self.params.wall_type == 'periodic':
            r = np.mod(r, L)
        else:
            # reflectivo: si una coordenada sale del rango [0, L], reflejar y aplicar restitución
            m = self.particles.get('m', np.ones(self.params.N, dtype=np.float64))
            e = float(self.params.restitution)
            for idx in range(self.params.N):
                for coord in range(3):
                    if r[idx, coord] < 0.0:
                        r[idx, coord] = -r[idx, coord]
                        v[idx, coord] = -e * v[idx, coord]
                    elif r[idx, coord] > L:
                        r[idx, coord] = 2.0 * L - r[idx, coord]
                        v[idx, coord] = -e * v[idx, coord]

        self.particles['r'] = r

        # 2. Guardar aceleración actual 
        a_old = a.copy() 

        # 3. Recalcular fuerzas 
        self.compute_forces()
        a_new = self.particles['a']

        # 4. Actualizar velocidades (v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt)) * dt)
        v += 0.5 * (a_old + a_new) * dt
        self.particles['v'] = v
        
        # 5. Control de Temperatura 
        if self.current_step < self.params.ign:
            E_kin = 0.5 * np.sum(v**2)
            current_T = 2.0 * E_kin / (3.0 * self.params.N)
            
            tau = 0.1
            scale_factor = np.sqrt(1.0 + (dt / tau) * ((self.params.T_0 / current_T) - 1.0))
            
            v *= scale_factor
            self.particles['v'] = v
        
        self.particles['a'] = a_new
        
    def do_step(self):
        """Ejecuta un paso de integración."""
        if not self.is_initialized:
            return {"error": "El simulador no ha sido inicializado."}
        
        try:
            self._integrate_step()
            self.current_step += 1
            
            self._compute_instantaneous_values()
            
            self._update_averages()

            return {
                "step": self.current_step,
                "r": self.particles['r'].tolist(), 
                "T": self.instant_values['T'],
                "E": self.instant_values['E_total'],
                "P": self.instant_values['P'],
                "Z": self.instant_values['Z'], 
                "is_equilibrium": self.current_step >= self.params.ign
            }
        except Exception as e:
             return {"error": f"Error en do_step: {str(e)}"}

    def get_averages(self):
        """Retorna los promedios calculados."""
        count = self.average_values['count']
        if count == 0:
            return {"T_avg": 0.0, "E_avg": 0.0, "P_avg": 0.0}
        
        return {
            "T_avg": self.average_values['T'] / count,
            "E_avg": self.average_values['E'] / count,
            "P_avg": self.average_values['P'] / count,
        }

# --- 2. CONFIGURACIÓN DE FLASK  ---
app = Flask(__name__, static_folder='static', static_url_path='/static')
simulator_instance = MolecularDynamicsSimulatorPython()

@app.route('/')
def index():
    return render_template('simulador.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_simulation():
    try:
        data = request.json
        data['N'] = int(data.get('N', 50))
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
        avg_T, avg_E, avg_P = simulator_instance.get_averages().values()
        return jsonify({
            "T_avg": f"{avg_T:.4f}", 
            "E_avg": f"{avg_E:.4f}", 
            "P_avg": f"{avg_P:.4f}"
        }), 200
    except Exception as e:
        print(f"Error al calcular promedios: {e}")
        return jsonify({"error": f"Error al calcular promedios: {str(e)}"}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "message": "pong"}), 200


@app.route('/api/particles', methods=['GET'])
def get_particles():
    if not simulator_instance.is_initialized:
        return jsonify({"error": "Simulación no inicializada."}), 400

    r = simulator_instance.particles['r']
    v = simulator_instance.particles['v']
    m = simulator_instance.particles['m']
    particles = []
    for i in range(r.shape[0]):
        particles.append({
            'index': i,
            'r': r[i].tolist(),
            'v': v[i].tolist(),
            'm': float(m[i])
        })
    return jsonify({'particles': particles}), 200


@app.route('/api/particles', methods=['POST'])
def add_particles():
    try:
        data = request.json
        new_particles = data.get('particles', [])
        if not new_particles:
            return jsonify({"error": "No se recibieron partículas."}), 400

        if not simulator_instance.is_initialized:
            return jsonify({"error": "Simulación no inicializada. Inicializa primero con /api/initialize."}), 400

        r = simulator_instance.particles['r'].tolist()
        v = simulator_instance.particles['v'].tolist()
        m = simulator_instance.particles['m'].tolist()

        for p in new_particles:
            r.append(p.get('r', [0.0, 0.0, 0.0]))
            v.append(p.get('v', [0.0, 0.0, 0.0]))
            m.append(float(p.get('m', simulator_instance.params.mass)))

        # Update particles and N
        simulator_instance.particles['r'] = np.array(r, dtype=np.float64)
        simulator_instance.particles['v'] = np.array(v, dtype=np.float64)
        simulator_instance.particles['m'] = np.array(m, dtype=np.float64)
        simulator_instance.particles['a'] = np.zeros((simulator_instance.particles['r'].shape[0], 3), dtype=np.float64)
        simulator_instance.params.N = simulator_instance.particles['r'].shape[0]

        # Recompute forces
        simulator_instance.compute_forces()

        return jsonify({"status": "ok", "N": simulator_instance.params.N}), 200
    except Exception as e:
        print(f"Error al añadir partículas: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/particles/<int:idx>', methods=['PUT'])
def update_particle(idx):
    try:
        if not simulator_instance.is_initialized:
            return jsonify({"error": "Simulación no inicializada."}), 400
        data = request.json
        r = simulator_instance.particles['r']
        v = simulator_instance.particles['v']
        m = simulator_instance.particles['m']

        if idx < 0 or idx >= r.shape[0]:
            return jsonify({"error": "Índice fuera de rango."}), 400

        if 'r' in data:
            r[idx] = np.array(data['r'], dtype=np.float64)
        if 'v' in data:
            v[idx] = np.array(data['v'], dtype=np.float64)
        if 'm' in data:
            m[idx] = float(data['m'])

        simulator_instance.particles['r'] = r
        simulator_instance.particles['v'] = v
        simulator_instance.particles['m'] = m
        simulator_instance.compute_forces()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"Error al actualizar partícula: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/particles/<int:idx>', methods=['DELETE'])
def delete_particle(idx):
    try:
        if not simulator_instance.is_initialized:
            return jsonify({"error": "Simulación no inicializada."}), 400

        r = simulator_instance.particles['r'].tolist()
        v = simulator_instance.particles['v'].tolist()
        m = simulator_instance.particles['m'].tolist()

        if idx < 0 or idx >= len(r):
            return jsonify({"error": "Índice fuera de rango."}), 400

        r.pop(idx)
        v.pop(idx)
        m.pop(idx)

        simulator_instance.particles['r'] = np.array(r, dtype=np.float64)
        simulator_instance.particles['v'] = np.array(v, dtype=np.float64)
        simulator_instance.particles['m'] = np.array(m, dtype=np.float64)
        simulator_instance.particles['a'] = np.zeros((simulator_instance.particles['r'].shape[0], 3), dtype=np.float64)
        simulator_instance.params.N = simulator_instance.particles['r'].shape[0]
        simulator_instance.compute_forces()

        return jsonify({"status": "ok", "N": simulator_instance.params.N}), 200
    except Exception as e:
        print(f"Error al borrar partícula: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/particles/import', methods=['POST'])
def import_particles_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se encontró archivo en la petición (campo 'file')."}), 400
        f = request.files['file']
        stream = io.StringIO(f.stream.read().decode('utf-8'))
        reader = csv.DictReader(stream)
        imported = []
        for row in reader:
      
            x = float(row.get('x', row.get('X', 0.0)))
            y = float(row.get('y', row.get('Y', 0.0)))
            z = float(row.get('z', row.get('Z', 0.0)))
            vx = float(row.get('vx', row.get('VX', 0.0)))
            vy = float(row.get('vy', row.get('VY', 0.0)))
            vz = float(row.get('vz', row.get('VZ', 0.0)))
            m = float(row.get('m', row.get('mass', simulator_instance.params.mass)))
            imported.append({'r': [x, y, z], 'v': [vx, vy, vz], 'm': m})

        params = {
            'N': len(imported),
            'NS': simulator_instance.params.NS,
            'dt': simulator_instance.params.dt,
            'T_0': simulator_instance.params.T_0,
            'ign': simulator_instance.params.ign,
            'bs': simulator_instance.params.bs,
            'sig': simulator_instance.params.sig,
            'eps': simulator_instance.params.eps,
            'mass': simulator_instance.params.mass,
            'wall_type': simulator_instance.params.wall_type,
            'restitution': simulator_instance.params.restitution,
            'particles': imported
        }

        state = simulator_instance.initialize(params)
        return jsonify({"status": "ok", "N": state['N']}), 200
    except Exception as e:
        print(f"Error al importar CSV: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/particles/export', methods=['GET'])
def export_particles_csv():
    try:
        if not simulator_instance.is_initialized:
            return jsonify({"error": "Simulación no inicializada."}), 400

        r = simulator_instance.particles['r']
        v = simulator_instance.particles['v']
        m = simulator_instance.particles['m']

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['x', 'y', 'z', 'vx', 'vy', 'vz', 'm'])
        for i in range(r.shape[0]):
            row = [float(r[i,0]), float(r[i,1]), float(r[i,2]), float(v[i,0]), float(v[i,1]), float(v[i,2]), float(m[i])]
            writer.writerow(row)

        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='particles_export.csv')
    except Exception as e:
        print(f"Error al exportar CSV: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)