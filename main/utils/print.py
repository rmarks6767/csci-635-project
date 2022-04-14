import sys
from data.data_loader import labels

def pretty_print_confusion(confusion):
  print('\n===== Confusion Matrix =====')
  for i in range(0, 20):
    sys.stdout.write(f'\t{i}')

  print('\n')

  label = 0
  for i in confusion:
    sys.stdout.write(f'{label}\t')
    for j in i:
      sys.stdout.write(f'{j}\t')

    label += 1
    print()
  print('\nLabels:')
  print('\t' + '\n\t'.join([f'{key} = {value}' for key, value in labels.items()]))
