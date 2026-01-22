# --------------------------
# 1. Compute Aid Effectiveness (per $10M)
# --------------------------
aid_effectiveness = {}

for country in all_countries:
    country_df = df[(df['country'] == country) & (df['year'] == 2022)]
    if country_df.empty:
        continue
    X_country_scaled = scaler.transform(country_df[features])

    # Collect marginal effects from all bootstraps
    effects = np.array([model.const_marginal_effect(X_country_scaled)[0] for model in bootstrap_models])
    # Average effect per country, scaled for $10M instead of $1M
    mean_effect = effects.mean() * 10
    aid_effectiveness[country] = mean_effect

# Convert to DataFrame
df_effectiveness = pd.DataFrame({
    'country': list(aid_effectiveness.keys()),
    'aid_effectiveness': list(aid_effectiveness.values())
})

# --------------------------
# 2. Dot plot: groups of 25 countries
# --------------------------
os.makedirs('aid_response_updated/dot_plots', exist_ok=True)

group_size = 25
num_groups = int(np.ceil(len(df_effectiveness) / group_size))

for i in range(num_groups):
    group_df = df_effectiveness.iloc[i*group_size:(i+1)*group_size]
    plt.figure(figsize=(14, 6))
    plt.scatter(group_df['country'], group_df['aid_effectiveness'], color='#1f77b4', s=100)

    # Add value labels on top of each dot
    for j, val in enumerate(group_df['aid_effectiveness']):
        plt.text(group_df['country'].iloc[j], val + 0.005, f"{val:.2f}",
                 ha='center', va='bottom', fontsize=9, rotation=0)

    plt.xticks(rotation=90)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('Aid Effectiveness (% ΔAccess per $10M)', fontsize=14)
    plt.title(f'Aid Effectiveness per Country (Group {i+1})', fontsize=16)
    plt.grid(alpha=0.2)

    # Increase upper margin to prevent overlap with top labels
    max_val = group_df['aid_effectiveness'].max()
    plt.ylim(0, max_val + 0.1 * max_val)  # add 10% extra space above the highest dot

    plt.tight_layout()
    plt.savefig(f'aid_response_updated/dot_plots/aid_effectiveness_group_{i+1}.png', dpi=300)
    plt.close()


# --------------------------
# 3. Horizontal bar chart: top/bottom countries
# --------------------------
os.makedirs('aid_response_updated/bar_charts', exist_ok=True)

# Sort countries
df_sorted = df_effectiveness.sort_values('aid_effectiveness', ascending=False)

# Top 10 highest
top_df = df_sorted.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_df['country'][::-1], top_df['aid_effectiveness'][::-1], color='green')
plt.xlabel('Aid Effectiveness (% ΔAccess per $10M)', fontsize=14)
plt.title('Top 10 Countries by Aid Effectiveness', fontsize=16)
plt.tight_layout()
plt.savefig('aid_response_updated/bar_charts/top10_aid_effectiveness.png', dpi=300)
plt.close()

# Bottom 10 lowest
bottom_df = df_sorted.tail(10)
plt.figure(figsize=(10, 6))
plt.barh(bottom_df['country'][::-1], bottom_df['aid_effectiveness'][::-1], color='red')
plt.xlabel('Aid Effectiveness (% ΔAccess per $10M)', fontsize=14)
plt.title('Bottom 10 Countries by Aid Effectiveness', fontsize=16)
plt.tight_layout()
plt.savefig('aid_response_updated/bar_charts/bottom10_aid_effectiveness.png', dpi=300)
plt.close()
