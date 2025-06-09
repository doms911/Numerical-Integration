import numpy as np
import matplotlib.pyplot as plt

def runge_kutta4(A, B, x0, r, T, t_max):
    t = 0
    x = x0.astype(np.float64)
    results = [(t, x.copy())]
    while t < t_max and abs(t - t_max) > 1e-6:
        m1 = A @ x + B @ r(t)
        m2 = A @ (x + m1 * T/2) + B @ r(t + T/2)
        m3 = A @ (x + m2 * T/2) + B @ r(t + T/2)
        m4 = A @ (x + m3 * T) + B @ r(t + T)
        x += T/6 * (m1 + 2 * m2 + 2 * m3 + m4)
        t += T
        results.append((t, x.copy()))
    return results

def trapezoidal_method(A, B, x0, r, T, t_max):
    t = 0
    x = x0.astype(np.float64)
    results = [(t, x.copy())]
    I = np.eye(A.shape[0])          # jedinicna matrica
    inv_mat = np.linalg.inv(I - A * T/2)
    R = inv_mat @ (I + A * T / 2)
    S = (inv_mat @ (T/2 * B))
    while t < t_max and abs(t - t_max) > 1e-6:
        x = R @ x + S @ (r(t) + r(t + T))
        t += T
        results.append((t, x.copy()))
    return results


def euler_explicit(A, B, x0, r, T, t_max):
    t = 0
    x = x0.astype(np.float64)
    results = [(t, x.copy())]
    I = np.eye(A.shape[0])
    M = I + A * T
    N = T * B
    while t < t_max and abs(t - t_max) > 1e-6:
        x = M @ x + N @ r(t)
        t += T
        results.append((t, x.copy()))
    return results

def backward_euler(A, B, x0, r, T, t_max):
    t = 0
    x = x0.astype(np.float64)
    results = [(t, x.copy())]
    I = np.eye(A.shape[0])
    P = np.linalg.inv(I - A * T)
    Q = P @ (T * B)
    while t < t_max and abs(t - t_max) > 1e-6:
        x = P @ x + Q @ r(t + T)
        t += T
        results.append((t, x.copy()))
    return results

def predictor_corrector_method(A, B, x0, r, T, t_max, predictor, corrector, num_corrections):
    t = 0
    x = x0.astype(np.float64)
    results = [(t, x.copy())]
    while t < t_max and abs(t - t_max) > 1e-6:
        x_pred = predictor(A, B, x, r, T, t)
        x_corr = x_pred
        for _ in range(num_corrections):
            x_corr = corrector(A, B, x, r, T, t)
        x = x_corr
        t += T
        results.append((t, x.copy()))
    return results

def euler_predictor(A, B, x, r, T, t):
    return x + T * (A @ x + B @ r(t))

def runge_kutta_predictor(A, B, x, r, T, t):
    k1 = A @ x + B @ r(t)
    k2 = A @ (x + T / 2 * k1) + B @ r(t + T / 2)
    k3 = A @ (x + T / 2 * k2) + B @ r(t + T / 2)
    k4 = A @ (x + T * k3) + B @ r(t + T)
    return x + T / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def trapezoidal_corrector(A, B, x, r, T, t):
    I = np.eye(A.shape[0])
    inv_mat = np.linalg.inv(I - A * T / 2)
    R = inv_mat @ (I + A * T / 2)
    S = inv_mat @ (T / 2 * B)
    return R @ x + S @ (r(t) + r(t + T))

def backward_euler_corrector(A, B, x, r, T, t):
    I = np.eye(A.shape[0])
    inv_mat = np.linalg.inv(I - A * T)
    Q = inv_mat @ (T * B)
    return inv_mat @ x + Q @ r(t + T)

def load_matrix(filename):
 return np.loadtxt(filename)

def run_an_example(A, B, x0, r, T, t_max, step, num_task):
    print("##########################")
    print(f'ZADATAK {num_task}')
    print("##########################")
    result_dict = {}
    # Runge-Kutta 4
    rgk4 = runge_kutta4(A, B, x0, r, T, t_max)
    print_results("Runge-Kutta postupak 4. reda", rgk4, step)
    result_dict['Runge-Kutta 4'] = rgk4
    print('__________________________')

    # Trapezni
    trapez = trapezoidal_method(A, B, x0, r, T, t_max)
    print_results("Trapezni postupak", trapez, step)
    result_dict['Trapezni'] = trapez
    print('__________________________')

    # Euler
    euler = euler_explicit(A, B, x0, r, T, t_max)
    print_results("Eulerov postupak", euler, step)
    result_dict['Euler'] = euler
    print('__________________________')

    # Backwards Euler
    backwards_eulr = backward_euler(A, B, x0, r, T, t_max)
    print_results("Obrnuti Eulerov postupak", backwards_eulr, step)
    result_dict['BackwardEuler'] = backwards_eulr
    print('__________________________')

    # PE(CE)^2
    pece2 = predictor_corrector_method(A, B, x0, r, T, t_max, euler_predictor, backward_euler_corrector, 2)
    print_results("PE(CE)^2 postupak", pece2, step)
    result_dict['PE(CE)^2'] = pece2
    print('__________________________')

    # PECE
    pece = predictor_corrector_method(A, B, x0, r, T, t_max, euler_predictor, trapezoidal_corrector, 2)
    print_results("PECE postupak", pece, step)
    result_dict['PECE'] = pece
    print('__________________________')

    return result_dict

def print_results(name, results, step):
    idx = 0
    print(name)
    for res in results:
        if abs(res[0] - idx) < 1e-6:
            print(f't = {res[0]}, x = {res[1]}')
            idx += step

def plot_all_results(results_dict, title, task_num):
    plt.figure(figsize=(12, 8))

    for method, results in results_dict.items():
        times = [t for t, _ in results]
        values = np.array([x for _, x in results])

        for i in range(values.shape[1]):
            plt.plot(times, values[:, i], label=f"{method} - x{i + 1}")

    plt.title(title)
    plt.xlabel("Vrijeme")
    plt.ylabel("Vrijednost varijable")
    plt.legend()
    plt.grid(True)
    path = f'{task_num}/'
    plt.savefig(path + 'graf_x1x2_vs_t.jpg')
    plt.show()


def plot_x1_x2_comparison(results_dict, title, task_num):
    plt.figure(figsize=(10, 8))

    for method, results in results_dict.items():
        values = np.array([x for _, x in results])
        x1 = values[:, 0]
        x2 = values[:, 1]

        plt.plot(x1, x2, label=method)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    path = f'{task_num}/'
    plt.savefig(path + 'graf_x1_vs_x2.jpg')
    plt.show()

def cum_sum(results, analytical_solution):
    cum_err = np.array([0., 0.], dtype=np.float64)
    for t, x in results:
        x_anal = analytical_solution(t)
        err = np.abs(x - x_anal)
        cum_err += err
    return cum_err

def analytical_solution_1(t):
    x1 = np.cos(t) + np.sin(t)
    x2 = np.cos(t) - np.sin(t)
    return np.array([x1, x2])

def prvi():
    A = load_matrix('1/matrices/A.txt')
    B = load_matrix('1/matrices/B.txt')
    x0 = load_matrix('1/matrices/x0.txt')
    r = lambda t: np.array([0, 0], dtype=np.float64)
    T = 0.01
    t_max = 10
    result_dict = run_an_example(A, B, x0, r, T, t_max, 1, 1)

    for key in result_dict:
        err = cum_sum(result_dict[key], analytical_solution_1)
        print(key + f' - err_x1: {np.round(err[0], 7)}, err_x2: {np.round(err[1], 7)}')

    plot_all_results(result_dict, "Grafička usporedba svih metoda - zadatak 1", 1)
    plot_x1_x2_comparison(result_dict, title="Usporedba x1 i x2 - zadatak 1", task_num=1)


def drugi():
    T = 0.1
    t_max = 1
    A = load_matrix("2/matrices/A.txt")
    B = np.array([[0], [0]], dtype=np.float64)
    x0 = load_matrix("2/matrices/x0.txt")
    r = lambda t: np.array([0], dtype=np.float64)
    result_dict = run_an_example(A, B, x0, r, T, t_max, 0.1, 2)
    plot_all_results(result_dict, "Grafička usporedba svih metoda - zadatak 2", 2)
    plot_x1_x2_comparison(result_dict, title="Usporedba x1 i x2 - zadatak 2", task_num=2)


def treci():
    A = load_matrix("3/matrices/A.txt")
    B = load_matrix("3/matrices/B.txt")
    r_matrix = load_matrix("3/matrices/r.txt")
    r = lambda t: r_matrix
    x0 = load_matrix("3/matrices/x0.txt")
    T = 0.01
    t_max = 10
    result_dict = run_an_example(A, B, x0, r, T, t_max, 1, 3)
    plot_all_results(result_dict, "Grafička usporedba svih metoda - zadatak 3", 3)
    plot_x1_x2_comparison(result_dict, title="Usporedba x1 i x2 - zadatak 3", task_num=3)

def cetvrti():
    A = load_matrix("4/matrices/A.txt")
    B = load_matrix("4/matrices/B.txt")
    r = lambda t: np.array([t, t], dtype=np.float64)
    x0 = load_matrix("4/matrices/x0.txt")
    T = 0.01
    t_max = 1
    result_dict = run_an_example(A, B, x0, r, T, t_max, 0.1, 4)
    plot_all_results(result_dict, "Grafička usporedba svih metoda - zadatak 4", 4)
    plot_x1_x2_comparison(result_dict, title="Usporedba x1 i x2 - zadatak 4", task_num=4)

if __name__ == "__main__":
   prvi()
   drugi()
   treci()
   cetvrti()