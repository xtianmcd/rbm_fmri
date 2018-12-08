import matplotlib.pyplot as plt

with open('./training_outputs.txt', 'r') as train:
    outputs = train.read()

training=[]
testing=[]

for line in outputs.split('\n'):
    if line:
        training.append(float(line.split(':')[1].split(' ')[1].strip()))
        testing.append(float(line.split(':')[2].strip()))
print(training)

plt.plot(range(1,58501,500), training, label='training')
plt.plot(range(1,58501,500), testing, label='testing')
plt.legend(loc='best')
plt.show()
