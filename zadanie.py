import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# y - lista o 2 składowych: y[0] = y, a y[1] = v  
# warunki początkowe  y0, vo

def f_op(t,y,m,k): 
    return [y[1], -k/m*y[1]*np.abs(y[1])-g]

m = 50.0 # masa pocisku
k = 5 # współczynnik oporu ciała
g = 9.81 # przyspieszenie ziemskie
y_0 = 100 # wysokość poczatkowa
v_0 = 60  # prędkość początkowa
t_max = 25 #czas maksymalny
y40 = [y_0, v_0] # lista z warunkami początkowymi
t_ob = np.arange(0,t_max,0.0001)

# Warunek  na osiągnięcie gruntu  przez spadające ciało
def grunt(t,y):
    return y[0]

# solve_ivp() oblicza pierwiastki równania grunt(t,y)=0

# Przerwanie obliczeń po kontakcie z gruntem
grunt.terminal = True
w_op = solve_ivp(lambda t,y:f_op(t,y,m,k), [0,t_max], y40, events=grunt,atol=1e-12,rtol=1e-10)

# Kilka wyników prezentujących zmieniającą się Wysokość i Prędkość
print("\nWysokość:")
for i in range(0,21):
    print(f"{i+1} : {w_op.y[0][i]}")
print("\n")    
print("Prędkość:")
for i in range(0,21):
    print(f"{i+1} : {w_op.y[1][i]}")
print("\n")    
print("Czas:")    
for i in range(0,21):
    print(f"{i+1} : {w_op.t[i]}")          
    
tablica_y = []
tablica_v = []

for i in range(len(w_op.y[0])):
    if w_op.y[0][i] > 99.99999:
        tablica_y.append(w_op.y[0][i])
        tablica_v.append(w_op.y[1][i])
        
print(f"Prędkość początkowa wynosi: {max(w_op.y[1])}")    
print(f"Maksymalna wysokość na jaką wzniesie się ciało wynosi:{max(w_op.y[0])}")
print(f"Prędkość uderzenia dla h = 0 m wynosi: {w_op.y[1,-1]}")        
print(f"Prędkość upadku do poziomu h=100 wynosi: {tablica_v[-1]}")

# Rysuję wykres h(t)
plot1 = plt.figure(1)
plt.plot(w_op.t,w_op.y[0],linewidth=3,color='red',label = "Wykres h(t) ")
plt.title("Wysokość pocisku w czasie")
plt.axhline(y=100, color="black", label="Taras")
plt.xlabel("Czas (t)")
plt.ylabel("Wysokość (h)")
plt.legend()

# Rysuję wykres h(v)
plot2 = plt.figure(2)
plt.plot(w_op.y[1],w_op.y[0],linewidth=3,color='red',label = "Wykres h(v) ")
plt.plot(tablica_v[-1],100,"go",label="Prędkość uderzenia w taras")
plt.title("Wysokość pocisku w zależności od prędkości")
plt.xlabel("prędkość (v)")
plt.ylabel("Wysokość (h)")
plt.axhline(y=100, color="black", label="Taras")
plt.legend()
plt.show()