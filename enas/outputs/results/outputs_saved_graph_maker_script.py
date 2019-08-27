import re
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="The name of the ENAS output file to parse")
file_name = parser.parse_args().file_name

content = [re.findall(r'val_acc=(1.0+|0.\d+)', line) for line in open(file_name)]
content = [float(c[0]) for c in content if c]

avg_values = list()
max_values = list()

for epoch in zip(*(iter(content),) * 10):
    avg_values.append(sum(epoch) / len(epoch))
    max_values.append(max(epoch))

max_value_avg = max(avg_values)
indexes_max_values_avg = [i for i in range(len(avg_values)) if avg_values[i] == max_value_avg]

max_value_max = max(max_values)
indexes_max_values_max = [i for i in range(len(max_values)) if max_values[i] == max_value_max]

plt.subplot(2, 1, 1)
plt.plot(avg_values, "-", color="#202F3C", linewidth=1)
for max_point in indexes_max_values_avg:
    plt.plot(max_point, max_value_avg, 'x', color="#5B2C6F")
    plt.text(max_point, max_value_avg, '({}, {})'.format(max_point, round(max_value_avg, 4)), fontsize=7)
plt.ylabel("Average accuracy values")
plt.axis([0, 200, 0, 1])

plt.subplot(2, 1, 2)
plt.plot(max_values, "-", color="#9A7D0A", linewidth=1)
for max_point in indexes_max_values_max:
    plt.plot(max_point, max_value_max, 'x', color="#5B2C6F")
    plt.text(max_point, max_value_max, '({}, {})'.format(max_point, round(max_value_max, 4)), fontsize=7)
plt.ylabel("Average accuracy values")
plt.ylabel("Max accuracy values")
plt.xlabel("Epochs")
plt.axis([0, 200, 0, 1])

# plt.show()
plt.savefig('{}_graph.png'.format(file_name), dpi=400)
