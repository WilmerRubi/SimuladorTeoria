import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS


class SimulationParams:
    def __init__(self, params_dict):
        # Parámetros de entrada desde el front
        self.N = int(params_dict.get("N", 60))
        self.NS = int(params_dict.get("NS", 2000))  # número máximo de pasos
        self.dt = float(params_dict.get("dt", 0.002))
        self.T_0 = float(params_dict.get("T_0", 1.0))
        self.ign = int(params_dict.get("ign", 200))  # pasos a ignorar para promedios

        # Parámetros del potencial de Lennard-Jones
        self.sig = float(params_dict.get("sig", 1.0))
        self.eps = float(params_dict.get("eps", 1.0))
        self.r_ctf = float(params_dict.get("r_ctf", 2.5))

        # Tamaño de la caja (lado del cubo)
        # bs lo usamos directamente como L en este modelo
        self.bs = float(params_dict.get("bs", 8.0))
        self.L = self.bs
        self.Volume = self.L ** 3

        # Pre-cálculos para LJ
        self.rcut2 = self.r_ctf ** 2
        self.sig6 = self.sig ** 6
        self.sig12 = self.sig6 ** 2
        self.eps4 = 4.0 * self.eps


# -----------------------------
# Motor de Dinámica Molecular
# -----------------------------

class MolecularDynamicsSimulatorPython:
    def __init__(self):
        self.params = None
        self.particles = {}
        self.current_step = 0
        self.is_initialized = False

        self.instant_values = {
            "T": 0.0,
            "P": 0.0,
            "E_total": 0.0,
            "Z": 0.0
        }

        self.simulation_data = {
            "T": [],
            "E_total": [],
            "P": []
        }

    # ---------- Inicialización ----------

    def initialize(self, params_dict):
        self.params = SimulationParams(params_dict)
        N = self.params.N
        L = self.params.L

        # Posiciones, velocidades, aceleraciones
        self.particles["r"] = np.zeros((N, 3), dtype=np.float64)
        self.particles["v"] = np.zeros((N, 3), dtype=np.float64)
        self.particles["a"] = np.zeros((N, 3), dtype=np.float64)
        self.particles["a_old"] = np.zeros((N, 3), dtype=np.float64)

        # Colocar partículas en una red cúbica simple dentro de la caja
        n_side = int(np.ceil(N ** (1.0 / 3.0)))
        spacing = L / n_side
        idx = 0
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if idx >= N:
                        break
                    self.particles["r"][idx, :] = [
                        (ix + 0.5) * spacing,
                        (iy + 0.5) * spacing,
                        (iz + 0.5) * spacing,
                    ]
                    idx += 1
                if idx >= N:
                    break
            if idx >= N:
                break

        # Velocidades iniciales aleatorias (distribución normal)
        self.particles["v"] = np.random.normal(0.0, 1.0, size=(N, 3))

        # Quitar velocidad del centro de masa
        v_cm = np.mean(self.particles["v"], axis=0)
        self.particles["v"] -= v_cm

        # Calcular energía cinética inicial y reescalar a T_0
        KE_initial = 0.5 * np.sum(self.particles["v"] ** 2)
        dof = 3 * N  # grados de libertad
        if KE_initial > 0.0:
            T_inst = (2.0 * KE_initial) / dof
            scale = np.sqrt(self.params.T_0 / T_inst)
            self.particles["v"] *= scale

        # Calcular fuerzas y energía potencial inicial
        U_total, virial = self._calculate_forces_and_energy()

        # Guardar estado inicial
        KE = 0.5 * np.sum(self.particles["v"] ** 2)
        self._calculate_instantaneous_values(KE, U_total, virial)

        # Reset de contadores
        self.current_step = 0
        self.simulation_data = {
            "T": [],
            "E_total": [],
            "P": []
        }
        self.is_initialized = True

        return {
            "L": float(L),
            "N": int(N),
            "r": self.particles["r"].tolist(),  # Enviamos X, Y, Z completos
            "T": float(self.instant_values["T"]),
            "E": float(self.instant_values["E_total"]),
            "P": float(self.instant_values["P"]),
            "Z": float(self.instant_values["Z"])
        }

    # ---------- Utilidades internas ----------

    def _apply_pbc(self, dr):
        """
        Aplica condiciones periódicas usando imagen mínima.
        """
        L = self.params.L
        return dr - L * np.round(dr / L)

    def _calculate_forces_and_energy(self):
        """
        Calcula fuerzas Lennard-Jones y energía potencial total.
        """
        N = self.params.N
        r = self.particles["r"]
        L = self.params.L
        rcut2 = self.params.rcut2
        sig6 = self.params.sig6
        sig12 = self.params.sig12
        eps4 = self.params.eps4

        # Reset aceleraciones
        self.particles["a"].fill(0.0)

        U_total = 0.0
        virial = 0.0

        for i in range(N - 1):
            for j in range(i + 1, N):
                dr = r[i] - r[j]
                dr = self._apply_pbc(dr)
                r2 = np.dot(dr, dr)

                if r2 < rcut2 and r2 > 1e-12:
                    inv_r2 = 1.0 / r2
                    inv_r6 = (sig6 * (inv_r2 ** 3))
                    inv_r12 = sig12 * (inv_r2 ** 6)

                    # Energía Lennard-Jones
                    U_ij = eps4 * (inv_r12 - inv_r6)
                    U_total += U_ij

                    # Fuerza: F = -dU/dr
                    # |F| = 24 * eps * (2 * (sigma^12 / r^13) - (sigma^6 / r^7))
                    force_scalar = 24.0 * self.params.eps * (
                        2.0 * (sig12 * (inv_r2 ** 7)) - (sig6 * (inv_r2 ** 4))
                    )
                    fij = force_scalar * dr

                    self.particles["a"][i] += fij
                    self.particles["a"][j] -= fij

                    # Virial
                    virial += np.dot(dr, fij)

        return U_total, virial

    def _update_positions_velocities(self):
        """
        Integra un paso con Velocity Verlet.
        """
        dt = self.params.dt
        L = self.params.L
        r = self.particles["r"]
        v = self.particles["v"]
        a = self.particles["a"]

        # Guardar aceleraciones actuales
        self.particles["a_old"][:] = a

        # Actualizar posiciones
        r += v * dt + 0.5 * self.particles["a_old"] * (dt ** 2)

        # Aplicar PBC
        r[:] = np.mod(r, L)

        # Recalcular fuerzas y energía potencial
        U_total, virial = self._calculate_forces_and_energy()

        # Actualizar velocidades
        v += 0.5 * (self.particles["a_old"] + self.particles["a"]) * dt

        # Energía cinética
        KE = 0.5 * np.sum(v ** 2)

        return KE, U_total, virial

    def _calculate_instantaneous_values(self, KE, PE, virial):
        """
        Calcula T, P, E_total, Z instantáneos.
        """
        N = self.params.N
        V = self.params.Volume
        dof = 3 * N

        T = (2.0 * KE) / dof
        P = (N * T / V) - (virial / (3.0 * V))
        E_total = KE + PE

        if abs(T) > 1e-12:
            Z = (P * V) / (N * T)
        else:
            Z = 0.0

        self.instant_values["T"] = T
        self.instant_values["P"] = P
        self.instant_values["E_total"] = E_total
        self.instant_values["Z"] = Z

    # ---------- API pública ----------

    def do_step(self):
        if not self.is_initialized or self.params is None:
            return {"error": "La simulación no ha sido inicializada."}

        if self.current_step >= self.params.NS:
            return {"error": "Se alcanzó el número máximo de pasos."}

        self.current_step += 1

        KE, U_total, virial = self._update_positions_velocities()
        self._calculate_instantaneous_values(KE, U_total, virial)

        # Acumulación para promedios después de 'ign' pasos
        if self.current_step >= self.params.ign:
            self.simulation_data["T"].append(self.instant_values["T"])
            self.simulation_data["E_total"].append(self.instant_values["E_total"])
            self.simulation_data["P"].append(self.instant_values["P"])

        return {
            "step": int(self.current_step),
            "r": self.particles["r"].tolist(),  # posiciones 3D
            "T": float(self.instant_values["T"]),
            "E": float(self.instant_values["E_total"]),
            "P": float(self.instant_values["P"]),
            "Z": float(self.instant_values["Z"]),
            "is_equilibrium": self.current_step >= self.params.ign
        }

    def calculate_averages(self):
        if len(self.simulation_data["T"]) == 0:
            return {
                "avg_T": 0.0,
                "avg_E": 0.0,
                "avg_P": 0.0
            }

        avg_T = float(np.mean(self.simulation_data["T"]))
        avg_E = float(np.mean(self.simulation_data["E_total"]))
        avg_P = float(np.mean(self.simulation_data["P"]))

        return {
            "avg_T": avg_T,
            "avg_E": avg_E,
            "avg_P": avg_P
        }


# -----------------------------
# Flask App
# -----------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

simulator_instance = MolecularDynamicsSimulatorPython()


@app.route("/")
def index():
    return render_template("simulador.html")


@app.route("/api/initialize", methods=["POST"])
def api_initialize():
    data = request.get_json() or {}

    try:
        sim_data = simulator_instance.initialize(data)
        return jsonify(sim_data), 200
    except Exception as e:
        return jsonify({"error": f"Error al inicializar: {str(e)}"}), 400


@app.route("/api/step", methods=["POST"])
def api_step():
    result = simulator_instance.do_step()

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result), 200


@app.route("/api/averages", methods=["GET"])
def api_averages():
    avg = simulator_instance.calculate_averages()
    return jsonify(avg), 200


if __name__ == "__main__":
    print("Iniciando servidor Flask...")
    print("Abre tu navegador en: http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=True)
