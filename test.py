in1 = input().split()
for i in range(len(in1)):
    in1[i] = str(in1[i])

in2 = input().split()
for i in range(len(in2)):
    in2[i] = str(in2[i])

seen = set(in1)

res = []
for item in in2:
    if item in seen:
        res.append(item)

if len(res) == 0:
    print("NULL")
else:
