import numpy as np
import scipy.optimize as optimize

from qiskit import Aer, execute, QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.opflow import I, X, Y, Z
from qiskit.opflow import CircuitOp, SummedOp, StateFn
from qiskit.opflow import PauliExpectation, PauliTrotterEvolution, Suzuki
from qiskit.quantum_info import Statevector


def get_heisenberg_int(nb_qubits, j, periodic=True):

    local_terms = []
    for op in [X, Y, Z]:
        for i in range(nb_qubits-1):
            local_terms.append((I^i)^(op^2)^(I^(nb_qubits - i - 2)))
        if periodic:
            local_terms.append(op^(I^(nb_qubits - 2))^op)

    return j*SummedOp(local_terms)


def get_compact_2_qubits_unitary(param):
    """
    Voir https://journals.aps.org/pra/pdf/10.1103/PhysRevA.69.032315
    """
    b = np.pi / 2 - 2 * param
    a = 2 * param - np.pi / 2

    qc = QuantumCircuit(2)
    qc.rz(np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(a, 0)
    qc.ry(b, 1)
    qc.cx(0, 1)
    qc.ry(a, 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)

    return qc


def get_ansatz(nb_qubits, depth):
    nb_odd_params_per_layer = nb_qubits // 2  # following naming convention from paper,
    # so odd is for entangling between qubits {(0,1),(2,3),...} :|
    nb_even_params_per_layer = (nb_qubits - 1) // 2
    theta_params = ParameterVector('t', depth * nb_odd_params_per_layer)
    phi_params = ParameterVector('p', depth * nb_even_params_per_layer)

    qc = QuantumCircuit(nb_qubits)
    for l in range(depth):
        for i in range(nb_odd_params_per_layer):
            qc.compose(get_compact_2_qubits_unitary(theta_params[l * nb_odd_params_per_layer + i]), [2 * i, 2 * i + 1],
                       inplace=True)
        # qc.barrier(range(nb_qubits))
        for i in range(nb_even_params_per_layer):
            qc.compose(get_compact_2_qubits_unitary(phi_params[l * nb_even_params_per_layer + i]),
                       [2 * i + 1, 2 * (i + 1)], inplace=True)
        # qc.barrier(range(nb_qubits))

    return qc, theta_params, phi_params


def get_neel_state(nb_qubits):
    qc = QuantumCircuit(nb_qubits)
    qc.x([2 * i + 1 for i in range(nb_qubits // 2)])

    return qc


def state_evolution(state, hamiltonian, tau, num_reps, trotter_order):
    assert (state.num_qubits == hamiltonian.num_qubits)
    t_param = Parameter('t')
    evolution_op = (t_param * hamiltonian).exp_i()
    trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=trotter_order, reps=num_reps)).convert(
        evolution_op @ CircuitOp(state))
    qc = trotterized_op.bind_parameters({t_param: tau})

    return qc


def get_current_sv(init_state, hamiltonian, current_t):
    hamiltonian_matrix = hamiltonian.to_matrix()
    eigval, eigvec = np.linalg.eigh(hamiltonian_matrix)
    u = eigvec
    u_dag = np.conj(eigvec).T

    sv_sim = Aer.get_backend('statevector_simulator')
    init_state_sv = execute(init_state, sv_sim).result().get_statevector()

    return u @ np.exp(-1.j * eigval.reshape((-1, 1)) * current_t) * u_dag @ np.asarray(init_state_sv).reshape((-1, 1))


def objective_fct(params, init_state, circuit, hamiltonian, current_t):
    global current_infidelity

    grounded_qc = init_state.compose(circuit).bind_parameters(params)
    infidelity = 1 - np.linalg.norm((~StateFn(get_current_sv(init_state, hamiltonian, current_t)) @ StateFn(grounded_qc)).eval()) ** 2

    try:
        current_infidelity.append(infidelity)
    except NameError:
        print('current_infidelity not defined in globals -- skipping')

    return infidelity


def callback(xk):
    global parameters_hist
    global itr_nb
    global infidelity_hist
    global current_infidelity

    print(itr_nb)
    itr_nb += 1
    parameters_hist.append(xk)
    infidelity_hist.append(current_infidelity[-1])
    current_infidelity.clear()


def optimize_params(init_state, anzats, hamiltonian, current_t, seed, num_iter, epsilon):
    globals()['parameters_hist'] = []
    globals()['infidelity_hist'] = []
    globals()['current_infidelity'] = []
    globals()['itr_nb'] = 0

    np.random.seed(seed)
    init_params = np.random.random(anzats.num_parameters)

    out = optimize.minimize(fun=objective_fct,
                            args=(init_state, anzats, hamiltonian, current_t),
                            x0=init_params,
                            method='L-BFGS-B',
                            callback=callback,
                            jac=None,
                            # gradient will be estimated using 2-point finite difference estimation with an absolute step size.
                            bounds=[(0, 2 * np.pi)] * anzats.num_parameters,
                            options={'maxiter': num_iter,
                                     'gtol': epsilon})

    out.x.dump(f'final_params_{anzats.num_qubits}_{seed}_{num_iter}_{current_t}.npy')
    print(out.fun)

    np.array(parameters_hist).dump(f'hist_params_{anzats.num_qubits}_{seed}_{num_iter}_{current_t}.npy')
    np.array(infidelity_hist).dump(f'hist_infidelity_{anzats.num_qubits}_{seed}_{num_iter}_{current_t}.npy')


seed = 267
nb_qubits = 8
coupling = 1

hamiltonian = get_heisenberg_int(nb_qubits, coupling)
init_state = get_neel_state(nb_qubits)

depth_l = 16  # Value used to generate Fig. 5(a)
anzats, t_params, p_params = get_ansatz(nb_qubits, depth_l)

epsilon = 1e-4  # optimization threshold for infidelity (1 - F)

current_t = 2.
num_iter = 1000

# optimize_params(init_state=init_state,
#                 anzats=anzats,
#                 hamiltonian=hamiltonian,
#                 current_t=current_t,
#                 seed=seed,num_iter=num_iter,
#                 epsilon=epsilon)

params_final = np.load('final_params_8_267_1000_2.0.npy', allow_pickle=True)
