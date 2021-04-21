import pickle
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
test = '0.309859	-0.456474	-0.096897	0	0	0	1	0	0	1	0	0	1	0	0	0	0	0	0	1	1	0	1	0	0	0	0	0'
t = test.split('\t')
op = list()
for i in t:
  op.append(float(i))
# print(op)
print(model.predict([op]))
