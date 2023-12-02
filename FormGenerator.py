import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt

categories = ['business', 'culture', 'education', 'health', 'politics', 'property', 'sport', 'technology', 'travel']
precisions = [0.90, 0.90, 0.88, 0.83, 0.89, 0.78, 0.99, 0.86, 0.64]
recalls = [0.95, 0.67, 0.68, 0.90, 0.94, 0.52, 0.98, 0.55, 0.84]
f1_scores = [0.93, 0.77, 0.76, 0.86, 0.92, 0.62, 0.98, 0.67, 0.72]

bar_width = 0.25
index = range(len(categories))

fig, ax = plt.subplots()
bar1 = ax.bar([i - bar_width for i in index], precisions, width=bar_width, label='Precision')
bar2 = ax.bar(index, recalls, width=bar_width, label='Recall')
bar3 = ax.bar([i + bar_width for i in index], f1_scores, width=bar_width, label='F1-Score')

ax.set_xlabel('Category')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall and F1-Score for Each Category')
ax.set_xticks([i for i in index])
ax.set_xticklabels(categories, rotation=45)
ax.legend()

plt.show()
exit()



# Sample data
class_counts = [1888, 17497, 565, 623, 2134, 565, 1184, 693, 16004]
class_labels = ['health', 'business', 'politics', 'culture', 'property', 'education', 'travel', 'technology', 'sport']

# Set the style
sns.set(style='whitegrid')

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=class_labels, y=class_counts)

# Adding title and labels
plt.title('Distribution of Classes')
plt.xlabel('Type of News')
plt.ylabel('Counts')

# Show the plot
plt.show()
