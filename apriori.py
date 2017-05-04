def powerSet(items):
	N = len(items)
	for i in range(2**N):
		combo = []
		for j in range(N):
			if (i >> j) % 2 == 1:
				combo.append(items[j])
