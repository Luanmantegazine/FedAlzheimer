import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class FLMetricsValidator:

    def __init__(self, phase1_path: str, phase2_path: str):
        self.df1 = pd.read_csv(phase1_path)
        self.df2 = pd.read_csv(phase2_path)
        self.keys = ['Método', 'Modelo', 'Batch', 'Clientes']
        self.df1 = self.df1.sort_values(self.keys).reset_index(drop=True)
        self.df2 = self.df2.sort_values(self.keys).reset_index(drop=True)
        self._validate_structure()

    def _validate_structure(self):
        if not (self.df1[self.keys] == self.df2[self.keys]).all().all():
            raise ValueError("Chaves não batem entre Fase 1 e Fase 2")
        print("✅ Estrutura validada: chaves consistentes entre fases\n")

    @staticmethod
    def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]

        bh = ranked * m / (np.arange(1, m + 1))
        bh = np.minimum.accumulate(bh[::-1])[::-1]

        qvals = np.empty_like(bh)
        qvals[order] = bh
        return qvals

    def detect_outliers(self, df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
        outliers = []

        for col in metric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            mask = (df[col] < lower) | (df[col] > upper)

            if mask.any():
                outlier_rows = df[mask].copy()
                outlier_rows['metric'] = col
                outlier_rows['z_score'] = np.abs(stats.zscore(df[col]))[mask]
                outliers.append(outlier_rows)

        return pd.concat(outliers, ignore_index=True) if outliers else pd.DataFrame()

    def validate_phase_consistency(self, phase_name: str, df: pd.DataFrame) -> Dict:
        metrics_mu = [c for c in df.columns if c.endswith('μ') and c not in self.keys]
        metrics_sigma = [c for c in df.columns if c.endswith('σ') and c not in self.keys]

        results = {
            'high_variance': [],
            'suspicious_values': [],
            'cv_stats': []
        }

        for mu_col, sigma_col in zip(metrics_mu, metrics_sigma):
            metric_name = mu_col.split(' ')[0]

            # Coeficiente de variação (CV = σ/μ)
            cv = df[sigma_col] / df[mu_col]
            cv_mean = cv.mean()
            cv_max = cv.max()

            # Identificar alta variância (CV > 0.1 para métricas de acurácia)
            if metric_name in ['ACC', 'AUC', 'F1'] and cv_mean > 0.10:
                results['high_variance'].append({
                    'metric': metric_name,
                    'cv_mean': cv_mean,
                    'cv_max': cv_max,
                    'configs_affected': (cv > 0.10).sum()
                })

            # Valores suspeitos
            suspicious = df[
                ((df[mu_col] < 50) & (metric_name in ['ACC', 'AUC', 'F1'])) |
                (df[mu_col] > 100)
                ]

            if not suspicious.empty:
                for _, row in suspicious.iterrows():
                    results['suspicious_values'].append({
                        'metric': metric_name,
                        'value': row[mu_col],
                        'config': f"{row['Modelo']}+{row['Método']}+{row['Batch']}+{row['Clientes']}c"
                    })

            results['cv_stats'].append({
                'metric': metric_name,
                'cv_mean': cv_mean,
                'cv_std': cv.std(),
                'cv_min': cv.min(),
                'cv_max': cv_max
            })

        return results

    def compare_phases(self) -> pd.DataFrame:
        metrics_mu = [c for c in self.df1.columns if c.endswith('μ') and c not in self.keys]

        results = []
        p_raw = []

        for col in metrics_mu:
            base = col.split(' ')[0]

            x = self.df1[col].values
            y = self.df2[col].values
            diff = y - x

            n = len(diff)
            mean1, mean2 = x.mean(), y.mean()
            std1, std2 = x.std(ddof=1), y.std(ddof=1)
            mean_diff = diff.mean()
            std_diff = diff.std(ddof=1)

            sigma1 = self.df1[f"{base} σ"].mean()
            sigma2 = self.df2[f"{base} σ"].mean()

            # Testes de normalidade
            shapiro_W, shapiro_p = stats.shapiro(diff)

            # Teste t pareado
            t_stat, t_p = stats.ttest_rel(y, x)

            # Teste de Wilcoxon
            wilcoxon_W, wilcoxon_p = stats.wilcoxon(y, x, alternative='two-sided')

            # Cohen's d
            cohens_d = mean_diff / std_diff if std_diff != 0 else np.nan

            # Intervalo de confiança 95% para a diferença
            se = std_diff / np.sqrt(n)
            ci_lower = mean_diff - 1.96 * se
            ci_upper = mean_diff + 1.96 * se

            # Percentage improvement
            pct_improvement = (mean2 - mean1) / mean1 * 100 if mean1 != 0 else np.nan

            results.append({
                'metric': base,
                'n': n,
                'mean_phase1': mean1,
                'mean_phase2': mean2,
                'mean_diff': mean_diff,
                'pct_improvement': pct_improvement,
                'std_phase1': std1,
                'std_phase2': std2,
                'std_diff': std_diff,
                'sigma_phase1_mean': sigma1,
                'sigma_phase2_mean': sigma2,
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'shapiro_W': shapiro_W,
                'shapiro_p': shapiro_p,
                'normality': 'Yes' if shapiro_p > 0.05 else 'No',
                't_stat': t_stat,
                't_p': t_p,
                'wilcoxon_W': wilcoxon_W,
                'wilcoxon_p': wilcoxon_p,
                'cohens_d': cohens_d,
                'effect_size': self._interpret_cohens_d(cohens_d)
            })

            p_raw.append(t_p)

        # Correção FDR
        qvals = self.benjamini_hochberg(p_raw)
        for res, q in zip(results, qvals):
            res['q_fdr'] = q
            res['significant_fdr'] = '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else 'ns'

        return pd.DataFrame(results)

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        if np.isnan(d):
            return 'N/A'
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def compare_algorithms(self, df: pd.DataFrame, phase_name: str) -> pd.DataFrame:
        metrics_mu = [c for c in df.columns if c.endswith('μ') and c not in self.keys]

        algorithms = df['Método'].unique()
        results = []

        for metric_col in metrics_mu:
            metric_name = metric_col.split(' ')[0]

            # Agrupar por algoritmo
            groups = [df[df['Método'] == alg][metric_col].values for alg in algorithms]

            # ANOVA
            f_stat, anova_p = stats.f_oneway(*groups)

            # Kruskal-Wallis (não paramétrico)
            h_stat, kruskal_p = stats.kruskal(*groups)

            # Pairwise comparisons (post-hoc)
            pairwise = []
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms):
                    if i < j:
                        g1 = df[df['Método'] == alg1][metric_col].values
                        g2 = df[df['Método'] == alg2][metric_col].values

                        t_stat, t_p = stats.ttest_ind(g1, g2)
                        u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')

                        pairwise.append({
                            'comparison': f"{alg1} vs {alg2}",
                            'mean_diff': g1.mean() - g2.mean(),
                            't_p': t_p,
                            'u_p': u_p
                        })

            results.append({
                'phase': phase_name,
                'metric': metric_name,
                'f_stat': f_stat,
                'anova_p': anova_p,
                'h_stat': h_stat,
                'kruskal_p': kruskal_p,
                'significant': 'Yes' if anova_p < 0.05 else 'No',
                'pairwise': pairwise
            })

        return pd.DataFrame(results)

    def check_table_consistency(self) -> List[Dict]:
        discrepancies = []

        table_6_2 = [
            {'FL': 'FedAvg', 'Batch': 64, 'Rede': 'DenseNet-169', 'Clientes': 3, 'ACC_reported': 86.97},
            {'FL': 'FedProx', 'Batch': 32, 'Rede': 'ResNet-50', 'Clientes': 3, 'ACC_reported': 85.32},
            {'FL': 'FedProx', 'Batch': 64, 'Rede': 'DenseNet-169', 'Clientes': 5, 'ACC_reported': 85.00},
        ]

        for entry in table_6_2:
            mask = (
                    (self.df1['Método'] == entry['FL']) &
                    (self.df1['Batch'] == entry['Batch']) &
                    (self.df1['Modelo'] == entry['Rede']) &
                    (self.df1['Clientes'] == entry['Clientes'])
            )

            if mask.any():
                calculated = self.df1.loc[mask, 'ACC μ'].values[0]
                reported = entry['ACC_reported']
                diff = abs(calculated - reported)

                if diff > 0.01:  # Tolerância de 0.01%
                    discrepancies.append({
                        'table': 'Tabela 6.2',
                        'config': f"{entry['Rede']} + {entry['FL']} + {entry['Batch']} + {entry['Clientes']}c",
                        'reported': reported,
                        'calculated': calculated,
                        'difference': diff,
                        'status': ' X - DISCREPANCY'
                    })
                else:
                    discrepancies.append({
                        'table': 'Tabela 6.2',
                        'config': f"{entry['Rede']} + {entry['FL']} + {entry['Batch']} + {entry['Clientes']}c",
                        'reported': reported,
                        'calculated': calculated,
                        'difference': diff,
                        'status': 'OK'
                    })

        # Valores da Tabela 6.5 (melhores resultados Fase II)
        table_6_5 = [
            {'FL': 'FedProx', 'Batch': 32, 'Rede': 'ResNet-50', 'Clientes': 3, 'ACC_reported': 99.26},
            {'FL': 'FedAvg', 'Batch': 32, 'Rede': 'DenseNet-169', 'Clientes': 3, 'ACC_reported': 98.90},
        ]

        for entry in table_6_5:
            mask = (
                    (self.df2['Método'] == entry['FL']) &
                    (self.df2['Batch'] == entry['Batch']) &
                    (self.df2['Modelo'] == entry['Rede']) &
                    (self.df2['Clientes'] == entry['Clientes'])
            )

            if mask.any():
                calculated = self.df2.loc[mask, 'ACC μ'].values[0]
                reported = entry['ACC_reported']
                diff = abs(calculated - reported)

                if diff > 0.01:
                    discrepancies.append({
                        'table': 'Tabela 6.5',
                        'config': f"{entry['Rede']} + {entry['FL']} + {entry['Batch']} + {entry['Clientes']}c",
                        'reported': reported,
                        'calculated': calculated,
                        'difference': diff,
                        'status': 'X - DISCREPANCY'
                    })
                else:
                    discrepancies.append({
                        'table': 'Tabela 6.5',
                        'config': f"{entry['Rede']} + {entry['FL']} + {entry['Batch']} + {entry['Clientes']}c",
                        'reported': reported,
                        'calculated': calculated,
                        'difference': diff,
                        'status': 'OK'
                    })

        return discrepancies

    def generate_report(self):
        print("=" * 80)
        print(" RELATÓRIO DE VALIDAÇÃO DE MÉTRICAS - FEDERATED LEARNING")
        print("=" * 80)
        print()

        print(" 1. VALIDAÇÃO INTERNA - FASE 1")
        print("-" * 80)
        val1 = self.validate_phase_consistency("Fase 1", self.df1)

        if val1['high_variance']:
            print("\nALTA VARIÂNCIA DETECTADA:")
            for item in val1['high_variance']:
                print(f"  • {item['metric']}: CV médio = {item['cv_mean']:.4f}, "
                      f"{item['configs_affected']} configs afetadas")
        else:
            print(" Variância aceitável em todas as métricas")

        if val1['suspicious_values']:
            print("\n VALORES SUSPEITOS:")
            for item in val1['suspicious_values']:
                print(f"  • {item['metric']} = {item['value']:.2f}% em {item['config']}")
        else:
            print(" Todos os valores dentro do esperado")

        print("\n Estatísticas de Coeficiente de Variação (Fase 1):")
        cv_df1 = pd.DataFrame(val1['cv_stats'])
        print(cv_df1.to_string(index=False))

        print("\n" + "=" * 80)
        print(" 2. VALIDAÇÃO INTERNA - FASE 2")
        print("-" * 80)
        val2 = self.validate_phase_consistency("Fase 2", self.df2)

        if val2['high_variance']:
            print("\n  ALTA VARIÂNCIA DETECTADA:")
            for item in val2['high_variance']:
                print(f"  • {item['metric']}: CV médio = {item['cv_mean']:.4f}")
        else:
            print(" Variância aceitável em todas as métricas")

        if val2['suspicious_values']:
            print("\n VALORES SUSPEITOS:")
            for item in val2['suspicious_values']:
                print(f"  • {item['metric']} = {item['value']:.2f}% em {item['config']}")
        else:
            print(" Todos os valores dentro do esperado")

        print("\n Estatísticas de Coeficiente de Variação (Fase 2):")
        cv_df2 = pd.DataFrame(val2['cv_stats'])
        print(cv_df2.to_string(index=False))

        # 2. Comparação entre fases
        print("\n" + "=" * 80)
        print(" 3. COMPARAÇÃO FASE 1 vs FASE 2")
        print("-" * 80)
        comp = self.compare_phases()

        print("\n Resumo Estatístico:")
        summary_cols = ['metric', 'mean_phase1', 'mean_phase2', 'pct_improvement',
                        'cohens_d', 'effect_size', 't_p', 'significant_fdr']
        print(comp[summary_cols].to_string(index=False))

        print("\n Intervalos de Confiança 95%:")
        ci_cols = ['metric', 'mean_diff', 'ci_95_lower', 'ci_95_upper']
        print(comp[ci_cols].to_string(index=False))

        # 3. Comparação entre algoritmos
        print("\n" + "=" * 80)
        print(" 4. COMPARAÇÃO ENTRE ALGORITMOS")
        print("-" * 80)

        algo_comp1 = self.compare_algorithms(self.df1, "Fase 1")
        algo_comp2 = self.compare_algorithms(self.df2, "Fase 2")

        print("\n Fase 1 - ANOVA entre algoritmos:")
        print(algo_comp1[['metric', 'f_stat', 'anova_p', 'significant']].to_string(index=False))

        print("\n🔍 Fase 2 - ANOVA entre algoritmos:")
        print(algo_comp2[['metric', 'f_stat', 'anova_p', 'significant']].to_string(index=False))

        # 4. Verificação de consistência com tabelas
        print("\n" + "=" * 80)
        print(" 5. VERIFICAÇÃO DE CONSISTÊNCIA COM TABELAS DA DISSERTAÇÃO")
        print("-" * 80)

        discrepancies = self.check_table_consistency()
        disc_df = pd.DataFrame(discrepancies)

        if not disc_df.empty:
            print("\n" + disc_df.to_string(index=False))

            errors = disc_df[disc_df['status'].str.contains('x')]
            if not errors.empty:
                print(f"\n TOTAL DE DISCREPÂNCIAS: {len(errors)}")
            else:
                print("\n TODAS AS TABELAS CONSISTENTES")

        # 5. Outliers
        print("\n" + "=" * 80)
        print(" 6. DETECÇÃO DE OUTLIERS")
        print("-" * 80)

        metrics_mu = [c for c in self.df1.columns if c.endswith('μ') and c not in self.keys]

        outliers1 = self.detect_outliers(self.df1, metrics_mu)
        if not outliers1.empty:
            print("\n  Outliers detectados na Fase 1:")
            print(f"  Total: {len(outliers1)} configurações")
            print(outliers1[self.keys + ['metric', 'z_score']].to_string(index=False))
        else:
            print("\n Nenhum outlier detectado na Fase 1")

        outliers2 = self.detect_outliers(self.df2, metrics_mu)
        if not outliers2.empty:
            print("\n  Outliers detectados na Fase 2:")
            print(f"  Total: {len(outliers2)} configurações")
            print(outliers2[self.keys + ['metric', 'z_score']].to_string(index=False))
        else:
            print("\n Nenhum outlier detectado na Fase 2")

        print("\n" + "=" * 80)
        print(" FIM DO RELATÓRIO")
        print("=" * 80)

        return {
            'phase_comparison': comp,
            'algo_comparison_phase1': algo_comp1,
            'algo_comparison_phase2': algo_comp2,
            'table_consistency': disc_df,
            'outliers_phase1': outliers1,
            'outliers_phase2': outliers2
        }


def main():
    phase1_path = "resultados_metricas_fase1 - Fase 1.csv"
    phase2_path = "resultados_metricas_fase2 - Fase 2.csv"

    # Criar validador
    validator = FLMetricsValidator(phase1_path, phase2_path)

    # Gerar relatório completo
    results = validator.generate_report()

    # Salvar resultados em CSV
    results['phase_comparison'].to_csv('validation_phase_comparison.csv', index=False)
    results['algo_comparison_phase1'].to_csv('validation_algo_phase1.csv', index=False)
    results['algo_comparison_phase2'].to_csv('validation_algo_phase2.csv', index=False)

    if not results['table_consistency'].empty:
        results['table_consistency'].to_csv('validation_table_consistency.csv', index=False)

    print("\n Resultados salvos em:")
    print("  • validation_phase_comparison.csv")
    print("  • validation_algo_phase1.csv")
    print("  • validation_algo_phase2.csv")
    print("  • validation_table_consistency.csv")


if __name__ == "__main__":
    main()