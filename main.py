import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange

class ElevationProfileInterpolator:
    def __init__(self):
        self.data = {}
        self.results = {}

    def load_data(self, file_path: str, route_name: str) -> bool:
        df = pd.read_csv(file_path)

        if len(df.columns) < 2:
            print(f"Błąd: Plik {file_path} nie zawiera wystarczających kolumn")
            return False

        distance_col = df.columns[0]
        elevation_col = df.columns[1]

        distance = pd.to_numeric(df[distance_col], errors='coerce')
        elevation = pd.to_numeric(df[elevation_col], errors='coerce')

        valid_mask = ~(distance.isna() | elevation.isna())
        distance = distance[valid_mask].values
        elevation = elevation[valid_mask].values

        sort_idx = np.argsort(distance)
        distance = distance[sort_idx]
        elevation = elevation[sort_idx]

        self.data[route_name] = {
            'distance': distance,
            'elevation': elevation,
            'file_path': file_path
        }

        print(f"Załadowano trasę '{route_name}': {len(distance)} punktów")
        print(f"Dystans: {distance[0]:.2f} - {distance[-1]:.2f} m")
        print(f"Wysokość: {elevation.min():.1f} - {elevation.max():.1f} m")

        return True

    def normalize_domain(self, x: np.ndarray, target_range: Tuple[float, float] = (-1,1)) -> Tuple[np.ndarray, Dict]:
        x_min, x_max = x.min(), x.max()
        a, b = target_range
        x_norm = a + (b - a) * (x - x_min) / (x_max - x_min)

        transform_params = {
            'x_min': x_min,
            'x_max': x_max,
            'a': a,
            'b': b
        }

        return x_norm, transform_params

    def denormalize_domain(self, x_norm: np.ndarray, transform_params: Dict) -> np.ndarray:
        x_min = transform_params['x_min']
        x_max = transform_params['x_max']
        a = transform_params['a']
        b = transform_params['b']

        x = x_min + (x_max - x_min) * (x_norm - a) / (b - a)
        return x

    def lagrange_interpolation(self, x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        x_nodes_norm, transform_params = self.normalize_domain(x_nodes)
        x_eval_norm = self.normalize_domain(x_eval, (transform_params['a'], transform_params['b']))[0]

        def lagrange_basis(j, x):
            terms = [(x - x_nodes_norm[m]) / (x_nodes_norm[j] - x_nodes_norm[m])
                     for m in range(len(x_nodes_norm)) if m != j]
            return np.prod(terms, axis=0)

        y_eval = np.zeros_like(x_eval_norm, dtype=np.float64)

        for j in range(len(x_nodes_norm)):
            lj = np.array([lagrange_basis(j, x) for x in x_eval_norm])
            y_eval += y_nodes[j] * lj

        return y_eval


    def cubic_spline_interpolation(self, x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
        y_eval = cs(x_eval)

        return y_eval

    def select_nodes(self, distance: np.ndarray, elevation: np.ndarray, n_nodes: int, method: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        if n_nodes >= len(distance):
            return distance, elevation

        indices = None
        if method == 'uniform':
            indices = np.linspace(0, len(distance) - 1, n_nodes, dtype=int)
        elif method == 'random':
            np.random.seed(42)
            indices = np.sort(np.random.choice(range(1, len(distance) - 1), n_nodes - 2, replace=False))
            indices = np.concatenate([[0], indices, [len(distance) - 1]])
        elif method == 'adaptive':
            if len(distance) < 3:
                indices = np.arange(len(distance))
            else:
                d2y = np.zeros(len(elevation))
                for i in range(1, len(elevation) - 1):
                    h1 = distance[i] - distance[i - 1]
                    h2 = distance[i + 1] - distance[i]
                    d2y[i] = abs(2 * ((elevation[i + 1] - elevation[i]) / h2 - (elevation[i] - elevation[i - 1]) / h1) / (h1 + h2))

                importance = d2y.copy()
                importance[0] = importance[-1] = np.max(importance)

                indices = np.argsort(importance)[-n_nodes:]
                indices = np.sort(indices)
        elif method == 'chebyshev':
            x_norm, transform_params = self.normalize_domain(distance)

            k = np.arange(n_nodes)
            cheb_nodes_norm = np.cos((2 * k + 1) * np.pi / (2 * n_nodes))

            cheb_nodes = self.denormalize_domain(cheb_nodes_norm, transform_params)

            indices = [np.abs(distance - x).argmin() for x in cheb_nodes]
            indices = np.unique(indices)

        return distance[indices], elevation[indices]

    def calculate_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        max_error = np.max(np.abs(y_true - y_pred))

        return {
            'MAE': mae,
            'RMSE': rmse,
            'Max_Error': max_error
        }

    def analyze_route(self, route_name: str, node_counts: List[int], methods: List[str] = ['uniform']) -> Dict:
        if route_name not in self.data:
            print(f"Brak danych dla trasy: {route_name}")
            return {}

        data = self.data[route_name]
        distance = data['distance']
        elevation = data['elevation']

        x_dense = np.linspace(distance[0], distance[-1], len(distance) * 3)

        results = {
            'route_name': route_name,
            'original_points': len(distance),
            'distance_range': (distance[0], distance[-1]),
            'elevation_range': (elevation.min(), elevation.max()),
            'analyses': {}
        }

        for method in methods:
            results['analyses'][method] = {}

            for n_nodes in node_counts:
                if n_nodes > len(distance):
                    continue

                x_nodes, y_nodes = self.select_nodes(distance, elevation, n_nodes, method)

                try:
                    y_lagrange = self.lagrange_interpolation(x_nodes, y_nodes, x_dense)
                    y_lagrange_orig = self.lagrange_interpolation(x_nodes, y_nodes, distance)
                    errors_lagrange = self.calculate_errors(elevation, y_lagrange_orig)
                    lagrange_success = True
                except Exception as e:
                    print(f"Błąd interpolacji Lagrange dla {n_nodes} węzłów: {e}")
                    y_lagrange = np.full_like(x_dense, np.nan)
                    errors_lagrange = {'MAE': np.inf, 'RMSE': np.inf, 'Max_Error': np.inf}
                    lagrange_success = False

                try:
                    y_spline = self.cubic_spline_interpolation(x_nodes, y_nodes, x_dense)
                    y_spline_orig = self.cubic_spline_interpolation(x_nodes, y_nodes, distance)
                    errors_spline = self.calculate_errors(elevation, y_spline_orig)
                    spline_success = True
                except Exception as e:
                    print(f"Błąd interpolacji spline dla {n_nodes} węzłów: {e}")
                    y_spline = np.full_like(x_dense, np.nan)
                    errors_spline = {'MAE': np.inf, 'RMSE': np.inf, 'Max_Error': np.inf}
                    spline_success = False

                results['analyses'][method][n_nodes] = {
                    'x_nodes': x_nodes,
                    'y_nodes': y_nodes,
                    'x_dense': x_dense,
                    'y_lagrange': y_lagrange,
                    'y_spline': y_spline,
                    'errors_lagrange': errors_lagrange,
                    'errors_spline': errors_spline,
                    'lagrange_success': lagrange_success,
                    'spline_success': spline_success
                }

        self.results[route_name] = results
        return results

    def plot_interpolation_comparison(self, route_name: str, n_nodes: int, method: str = 'uniform', save_fig: bool = False):
        if route_name not in self.results:
            print(f"Brak wyników dla trasy: {route_name}")
            return

        data = self.data[route_name]
        results = self.results[route_name]['analyses'][method][n_nodes]

        plt.figure(figsize=(12, 8))

        plt.plot(data['distance'], data['elevation'], 'k-', linewidth=0.8,
                 label=f'Dane oryginalne ({len(data["distance"])} punktów)', alpha=0.7)

        plt.plot(results['x_nodes'], results['y_nodes'], 'ro', markersize=8,
                 label=f'Węzły interpolacji ({n_nodes} punktów)')

        if results['lagrange_success']:
            plt.plot(results['x_dense'], results['y_lagrange'], 'b--', linewidth=2,
                     label=f'Lagrange (RMSE: {results["errors_lagrange"]["RMSE"]:.2f}m)')

        if results['spline_success']:
            plt.plot(results['x_dense'], results['y_spline'], 'g-', linewidth=2,
                     label=f'Funkcje sklejane (RMSE: {results["errors_spline"]["RMSE"]:.2f}m)')

        plt.xlabel('Dystans [m]')
        plt.ylabel('Wysokość [m n.p.m.]')
        plt.title(f'Interpolacja profilu wysokościowego - {route_name}\n'
                  f'Metoda wyboru węzłów: {method}, Liczba węzłów: {n_nodes}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_fig:
            plt.savefig(f'interpolation_{route_name}_{method}_{n_nodes}nodes.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_error_analysis(self, route_name: str, methods: List[str] = ['uniform'],
                            save_fig: bool = False):
        if route_name not in self.results:
            print(f"Brak wyników dla trasy: {route_name}")
            return

        plt.figure(figsize=(15, 5))

        for method in methods:
            if method not in self.results[route_name]['analyses']:
                continue

            analyses = self.results[route_name]['analyses'][method]
            node_counts = sorted(analyses.keys())

            rmse_lagrange = []
            rmse_spline = []
            mae_lagrange = []
            mae_spline = []
            max_error_lagrange = []
            max_error_spline = []

            for n_nodes in node_counts:
                result = analyses[n_nodes]
                rmse_lagrange.append(result['errors_lagrange']['RMSE'])
                rmse_spline.append(result['errors_spline']['RMSE'])
                mae_lagrange.append(result['errors_lagrange']['MAE'])
                mae_spline.append(result['errors_spline']['MAE'])
                max_error_lagrange.append(result['errors_lagrange']['Max_Error'])
                max_error_spline.append(result['errors_spline']['Max_Error'])

            # RMSE
            plt.subplot(1, 3, 1)
            plt.plot(node_counts, rmse_lagrange, 'b-o', label=f'Lagrange ({method})')
            plt.plot(node_counts, rmse_spline, 'g-s', label=f'Spline ({method})')
            plt.xlabel('Liczba węzłów')
            plt.ylabel('RMSE [m]')
            plt.title('Root Mean Square Error')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # MAE
            plt.subplot(1, 3, 2)
            plt.plot(node_counts, mae_lagrange, 'b-o', label=f'Lagrange ({method})')
            plt.plot(node_counts, mae_spline, 'g-s', label=f'Spline ({method})')
            plt.xlabel('Liczba węzłów')
            plt.ylabel('MAE [m]')
            plt.title('Mean Absolute Error')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Max Error
            plt.subplot(1, 3, 3)
            plt.plot(node_counts, max_error_lagrange, 'b-o', label=f'Lagrange ({method})')
            plt.plot(node_counts, max_error_spline, 'g-s', label=f'Spline ({method})')
            plt.xlabel('Liczba węzłów')
            plt.ylabel('Błąd maksymalny [m]')
            plt.title('Maksymalny błąd bezwzględny')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()

        plt.suptitle(f'Analiza błędów interpolacji - {route_name}', fontsize=14)
        plt.tight_layout()

        if save_fig:
            plt.savefig(f'error_analysis_{route_name}.png', dpi=300, bbox_inches='tight')

        plt.show()

    def generate_summary_report(self) -> str:
        report = ["=" * 60, "PODSUMOWANIE ANALIZY INTERPOLACJI PROFILÓW WYSOKOŚCIOWYCH", "=" * 60]

        for route_name, results in self.results.items():
            report.append(f"\nTRASA: {route_name}")
            report.append("-" * 40)
            report.append(f"Liczba oryginalnych punktów: {results['original_points']}")
            report.append(
                f"Zakres dystansu: {results['distance_range'][0]:.2f} - {results['distance_range'][1]:.2f} m")
            report.append(
                f"Zakres wysokości: {results['elevation_range'][0]:.1f} - {results['elevation_range'][1]:.1f} m")

            for method, analyses in results['analyses'].items():
                report.append(f"\nMetoda wyboru węzłów: {method}")

                best_lagrange = None
                best_spline = None
                min_rmse_lagrange = float('inf')
                min_rmse_spline = float('inf')

                for n_nodes, result in analyses.items():
                    if result['errors_lagrange']['RMSE'] < min_rmse_lagrange:
                        min_rmse_lagrange = result['errors_lagrange']['RMSE']
                        best_lagrange = (n_nodes, result['errors_lagrange'])

                    if result['errors_spline']['RMSE'] < min_rmse_spline:
                        min_rmse_spline = result['errors_spline']['RMSE']
                        best_spline = (n_nodes, result['errors_spline'])

                if best_lagrange:
                    report.append(f"Najlepszy wynik Lagrange: {best_lagrange[0]} węzłów, "
                                  f"RMSE: {best_lagrange[1]['RMSE']:.3f}m")

                if best_spline:
                    report.append(f"Najlepszy wynik Spline: {best_spline[0]} węzłów, "
                                  f"RMSE: {best_spline[1]['RMSE']:.3f}m")

        return "\n".join(report)
        
    def plot_interpolation_comparison_multi(self, route_name: str, node_counts: List[int], method: str = 'uniform', save_dir: str = None):
        if route_name not in self.results:
            print(f"Brak wyników dla trasy: {route_name}")
            return

        data = self.data[route_name]
        results = self.results[route_name]['analyses'][method]
        
        # Helper for plotting
        def plot_figure(interpolation_type: str):
            fig, axs = plt.subplots(2, 2, figsize=(16, 10))
            axs = axs.flatten()
            for idx, n_nodes in enumerate(node_counts):
                if n_nodes not in results:
                    axs[idx].set_title(f"Brak danych dla {n_nodes} węzłów")
                    axs[idx].axis('off')
                    continue

                r = results[n_nodes]

                axs[idx].plot(data['distance'], data['elevation'], 'k-', label='Oryginalne dane', linewidth=0.8)
                axs[idx].plot(r['x_nodes'], r['y_nodes'], 'ro', markersize=8, label='Węzły')

                if interpolation_type == 'lagrange':
                    if r['lagrange_success']:
                        axs[idx].plot(r['x_dense'], r['y_lagrange'], 'b--', label='Lagrange')
                    axs[idx].set_title(f'Lagrange - {n_nodes} węzłów')
                elif interpolation_type == 'spline':
                    if r['spline_success']:
                        axs[idx].plot(r['x_dense'], r['y_spline'], 'g--', label='Spline')
                    axs[idx].set_title(f'Spline - {n_nodes} węzłów')

                axs[idx].legend(fontsize=8)
                axs[idx].grid(True)
                axs[idx].set_xlabel("Dystans [m]")
                axs[idx].set_ylabel("Wysokość [m]")

            fig.tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{route_name}_{method}_{interpolation_type}.png"
                fig.savefig(os.path.join(save_dir, filename), dpi=300)
            else:
                plt.show()

        plot_figure('lagrange')
        plot_figure('spline')


if __name__ == "__main__":
    interpolator = ElevationProfileInterpolator()

    sample_routes = {
        "Hel Yeah": "data/Hel_yeah.csv",
        "Spacerniak Gdańsk": "data/SpacerniakGdansk.csv"
    }

    print("Próba załadowania przykładowych danych...")

    loaded_routes = []
    for route_name, file_path in sample_routes.items():
        if os.path.exists(file_path):
            if interpolator.load_data(file_path, route_name):
                loaded_routes.append(route_name)

    node_counts = [8, 16, 32, 64]
    methods = ['uniform', 'chebyshev']

    print(f"\nRozpoczynanie analizy dla tras: {loaded_routes}")

    for route_name in loaded_routes:
        print(f"\nAnalizuję trasę: {route_name}")
        results = interpolator.analyze_route(route_name, node_counts, methods)
        
        print(f"\nTworzę wykresy trasy: {route_name}")
        for method in methods:
            interpolator.plot_interpolation_comparison_multi(route_name, node_counts, method, save_dir="assets")

    print("\n" + interpolator.generate_summary_report())

    print("\nAnaliza zakończona. Sprawdź wygenerowane wykresy powyżej.")
    print("\nAby użyć z własnymi danymi:")
    print("1. Umieść pliki CSV w odpowiednim katalogu")
    print("2. Zaktualizuj słownik 'sample_routes' z nazwami plików")
    print("3. Uruchom ponownie skrypt")
