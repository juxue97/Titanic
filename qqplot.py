import scipy.stats as stat
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(df, feature):
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'{feature} Histogram')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    # QQ Plot
    plt.subplot(1, 2, 2)
    (osm, osr), _ = stat.probplot(df[feature], dist='norm', plot=plt)
    plt.title(f'{feature} QQ Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

    # Assess QQ plot quality
    shapiro_test_stat, shapiro_p_value = stat.shapiro(df[feature])
    ks_test_stat, ks_p_value = stat.kstest((osm - osm.mean()) / osm.std(), 'norm')

    if shapiro_p_value < 0.05:
        shapiro_result = "Non-Normal"
    else:
        shapiro_result = "Normal"

    if ks_p_value < 0.05:
        ks_result = "Non-Normal"
    else:
        ks_result = "Normal"

    plt.text(0.1, 0.9, f'Shapiro-Wilk: {shapiro_result}\nKS Test: {ks_result}', 
             horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()