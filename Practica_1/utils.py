import matplotlib.pyplot as plt

def seteaGrilla():
	plt.grid()
	plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
	plt.minorticks_on()
	#plt.grid(b=True, color='#999999', linestyle='-', alpha=0.2)