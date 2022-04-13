import sys

def pretty_print_confusion(confusion):
  print('\n===== Confusion Matrix =====')
  for i in range(1, 21):
    sys.stdout.write(f'\t{i}')

  sys.stdout.write('\n\n')

  label = 1
  for i in confusion:
    sys.stdout.write(f'{label}\t')
    for j in i:
      sys.stdout.write(f'{j}\t')

    label += 1
    sys.stdout.write('\n')
  sys.stdout.write('\n')
  