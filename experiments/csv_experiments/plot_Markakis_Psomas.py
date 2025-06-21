import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# טעינת הנתונים
df = pd.read_csv('results.csv')

# חישוב ממוצעים וסטיות תקן
results = df.groupby(['algorithm', 'num_agents', 'num_items', 'distribution']).agg({
    'min_value': ['mean', 'std'],
    'sum_value': ['mean', 'std'],
    'execution_time': ['mean', 'std']
}).reset_index()

results.columns = ['_'.join(col).strip() if col[1] else col[0] for col in results.columns.values]
results.rename(columns={
    'algorithm_': 'algorithm',
    'num_agents_': 'num_agents',
    'num_items_': 'num_items',
    'distribution_': 'distribution'
}, inplace=True)

# הגדרת עיצוב הגרפים
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 1. השוואת אלגוריתמים לפי חלוקה ערכית
def plot_by_distribution(metric, ylabel):
    plt.figure(figsize=(12, 8))
    distributions = ['uniform', 'normal', 'exponential']
    
    for i, dist in enumerate(distributions, 1):
        plt.subplot(2, 2, i)
        dist_data = results[results['distribution'] == dist]
        
        sns.barplot(
            x='num_items', 
            y=f'{metric}_mean', 
            hue='algorithm',
            data=dist_data,
            palette=palette,
            edgecolor='black'
        )
        
        plt.title(f'Distribution: {dist.capitalize()}')
        plt.xlabel('Number of Items')
        plt.ylabel(ylabel)
        plt.legend(title='Algorithm', loc='upper left')
        plt.tight_layout()

    plt.suptitle(f'{ylabel} Comparison by Distribution', fontsize=16, y=1.02)
    plt.savefig(f'{metric}_by_distribution.png', bbox_inches='tight')
    plt.close()

# 2. השוואת אלגוריתמים לפי מספר סוכנים
def plot_by_agents(metric, ylabel):
    plt.figure(figsize=(14, 10))
    agents = sorted(df['num_agents'].unique())
    
    for i, agent in enumerate(agents, 1):
        plt.subplot(2, 2, i)
        agent_data = results[results['num_agents'] == agent]
        
        sns.lineplot(
            x='num_items', 
            y=f'{metric}_mean', 
            hue='algorithm',
            style='algorithm',
            data=agent_data,
            markers=True,
            dashes=False,
            markersize=10,
            palette=palette,
            linewidth=2.5
        )
        
        plt.title(f'{agent} Agents')
        plt.xlabel('Number of Items')
        plt.ylabel(ylabel)
        plt.xticks([10, 20, 50, 100])
        plt.grid(alpha=0.3)
        plt.legend(title='Algorithm', loc='best')
    
    plt.suptitle(f'{ylabel} Comparison by Number of Agents', fontsize=16, y=0.95)
    plt.savefig(f'{metric}_by_agents.png', bbox_inches='tight')
    plt.close()

# 3. ביצועים לפי גודל בעיה
def plot_performance():
    # יצירת משתנה גודל בעיה
    results['problem_size'] = results['num_agents'] * results['num_items']
    
    plt.figure(figsize=(12, 8))
    
    # זמן ריצה
    plt.subplot(2, 1, 1)
    your_algo = results[results['algorithm'] == 'Markakis&Posmas']
    sns.lineplot(
        x='problem_size', 
        y='execution_time_mean', 
        hue='distribution',
        style='distribution',
        data=your_algo,
        markers=True,
        dashes=False,
        markersize=8,
        linewidth=2
    )
    plt.title('Markakis&Posmas Execution Time')
    plt.xlabel('Problem Size (Agents × Items)')
    plt.ylabel('Execution Time (s)')
    plt.grid(alpha=0.3)
    plt.legend(title='Distribution')
    
    # השוואת ערך מינימלי
    plt.subplot(2, 1, 2)
    sns.lineplot(
        x='problem_size', 
        y='min_value_mean', 
        hue='algorithm',
        style='algorithm',
        data=results,
        markers=True,
        dashes=False,
        markersize=8,
        linewidth=2
    )
    plt.title('Minimum Value Comparison')
    plt.xlabel('Problem Size (Agents × Items)')
    plt.ylabel('Average Min Value')
    plt.grid(alpha=0.3)
    plt.legend(title='Algorithm', loc='lower right')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', bbox_inches='tight')
    plt.close()

# 4. יחס בין אלגוריתמים
def plot_ratios():
    # יצירת עותק של הנתונים
    ratio_df = results.copy()
    
    # חישוב יחסים לעומת RoundRobin
    for metric in ['min_value', 'sum_value']:
        ratio_df[f'{metric}_ratio'] = np.nan
        
        for _, row in ratio_df.iterrows():
            agent = row['num_agents']
            item = row['num_items']
            dist = row['distribution']
            algo = row['algorithm']
            value = row[f'{metric}_mean']
            
            # מציאת ערך הייחוס (RoundRobin)
            ref_value = results[
                (results['algorithm'] == 'RoundRobin') &
                (results['num_agents'] == agent) &
                (results['num_items'] == item) &
                (results['distribution'] == dist)
            ][f'{metric}_mean'].values
            
            if len(ref_value) > 0:
                ratio_df.loc[_, f'{metric}_ratio'] = value / ref_value[0]
    
    # ציור גרף יחסים
    plt.figure(figsize=(12, 8))
    
    for i, algo in enumerate(['Markakis&Posmas', 'Sequential'], 1):
        plt.subplot(1, 2, i)
        algo_data = ratio_df[ratio_df['algorithm'] == algo]
        
        sns.boxplot(
            x='distribution',
            y='min_value_ratio',
            data=algo_data,
            palette=palette,
            showmeans=True,
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'}
        )
        
        plt.axhline(1, color='red', linestyle='--', alpha=0.7)
        plt.title(f'{algo} vs RoundRobin')
        plt.xlabel('Distribution')
        plt.ylabel('Min Value Ratio')
        plt.ylim(0.5, 2.0)
        plt.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Algorithm Performance Ratios Relative to RoundRobin', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('performance_ratios.png', bbox_inches='tight')
    plt.close()

plot_by_distribution('min_value', 'Minimum Value (Fairness)')
plot_by_distribution('sum_value', 'Total Value (Efficiency)')

plot_by_agents('min_value', 'Minimum Value (Fairness)')
plot_by_agents('sum_value', 'Total Value (Efficiency)')

plot_performance()
plot_ratios()

print("Finish")