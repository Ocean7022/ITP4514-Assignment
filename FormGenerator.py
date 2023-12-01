import matplotlib.pyplot as plt
import seaborn as sns

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
