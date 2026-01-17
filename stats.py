import pandas as pd
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
# run the following if you're sane and not running nix:
# pip install pandas numpy sklearn factor_analyzer matplotlib

df = pd.read_csv("EIGHT_FACTOR_SURVEY_WITHOUT_ID_COLUMN.csv")
print("STATISTICAL SUMMARY")
print("------------------")

LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
# -------------------------------------------------------------------
# General stats
# -------------------------------------------------------------------
"""
median = df.median()
median.name="Median"
mode=df.mode().iloc[0].squeeze()
mode.name="Mode"
"""
# not used for the particular use-case, uncomment it if you so need

num = df.count()
num.name = "# Data points"
mean=df.mean()
mean.name="Mean"
min = df.min()
min.name = "Min."
max=df.max()
max.name="Max."

std=df.std()
std.name= "Standard deviation"
cc = pd.concat([num, mean, std, #median, mode
    min, max], axis=1).transpose()
pd.set_option("display.precision", 4)
print(cc)

print("\n==========================================================\n")
print("Frequencies:")
molten = df.melt(var_name='columns', value_name='index')
print(pd.crosstab(index=molten['index'], columns=molten['columns']))

print("\n==========================================================\n")
print("Correlations:")
print(df.corr())

print("\n==========================================================\n")
kmo_all, kmo_model = calculate_kmo(df)
print(f"KMO: {kmo_model}")
print("- Per variable (Measurement System Analysis):")
kmo = pd.Series(kmo_all,index=LABELS)
print(kmo.to_string())

print("\n==========================================================\n")
# -------------------------------------------------------------------
# Relevancy
# -------------------------------------------------------------------
chi_sq, p = calculate_bartlett_sphericity(df)
print("Bartlett Sphericity:")
print(f"- χ² (chi-cuadrada): {chi_sq}")
print(f"- p-valor: {p}, {"reject" if p < 0.001 else "plausible"} H₀")

print("\n==========================================================\n")
print("Value extraction")
print("- PCA with Kaiser:")
stsc = StandardScaler()
bigX = stsc.fit_transform(df)
pca = PCA(n_components=8)
components = pca.fit_transform(bigX)
eigv = pd.Series(pca.explained_variance_, index=LABELS)
eigv.name="Eigenvalues"
varpc = pd.Series(pca.explained_variance_ratio_ * 100, index=LABELS)
varpc.name = "Proportion"
kscr = pd.Series(list(map(lambda a: "Retain" if a > 1 else "Discard", pca.explained_variance_)), index=LABELS, name="Kaiser")
print(pd.concat([eigv, varpc, kscr], axis=1))

print("\n- Skree plot")
print('-- will be shown once the other calculations are done. The user is to find the "bend" in the plot.') # in this particular case the order was scree first, then the other stuff. however pythreading scares me

PC_values = np.arange(pca.n_components_) + 1

print("\n- Varimax rotation matrix analysis")
fa = FactorAnalysis(rotation='varimax')
fa.fit(bigX)
varimax_comp = fa.components_.T
factores = pd.concat(list(map(lambda a: pd.Series(a), varimax_comp)), axis=1).transpose() # so glad these libraries had the specific functions. i would've gone insane trying to understand the algorithms in the 90 minutes i had left
factores['Variables'] = LABELS
factores.set_index('Variables', inplace=True)
print(factores)

print("\n==========================================================\n")
print("Per-factor ratings:")
# -------------------------------------------------------------------
# Per-(composite) factor ratings (simple mean)
# -------------------------------------------------------------------
df['F1'] = df[['V6', 'V7', 'V8']].mean(axis=1)
df['F2'] = df[['V1', 'V2', 'V5']].mean(axis=1)
df['F3'] = df[['V3', 'V4']].mean(axis=1)

# -------------------------------------------------------------------
# Statistical summary of ratings
# -------------------------------------------------------------------
summary = df[['F1', 'F2', 'F3']].agg(['mean', 'std', 'min', 'max', 'median'])
print("\nPER-VARIABLE DIGEST:\n")
print(summary)
print("\nFirst rows with per-factor ratings:\n")
print(df[['F1', 'F2', 'F3']].head())

fig, ax = plt.subplots(1,3)
fig.set_figheight(5)
fig.set_figwidth(13)
ax[0].plot(LABELS, mean, 'o-', linewidth=2, color="red")
ax[0].set_title('Promedios')
ax[0].set_xlabel('Variable')
ax[0].set_ylabel('Media')

ax[1].plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
ax[1].set_title('Scree Plot')
ax[1].set_xlabel('Principal Component')
ax[1].set_ylabel('Variance Explained')

medias = summary.loc['mean']
ax[2].bar(medias.index, medias.values)
ax[2].set_title("Comparativo de medias por factor")
ax[2].set_xlabel("Factores")
ax[2].set_ylabel("Media (1 a 5)")
ax[2].set_ylim(0, 5)   # Fiddling with scale so it's not terrible on the eyes *and* data

plt.show()
