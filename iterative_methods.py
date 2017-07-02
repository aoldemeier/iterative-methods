# coding=UTF8

import numpy as np
import matplotlib.pyplot as plt

# Problem
# (1) strictly diagonal dominant matrix
#A=np.matrix('0.7 -0.2 -0.1;-0.2 0.6 -0.1; -0.1 -0.1 0.9')
# (2) spd matrix
A=np.matrix('2.0 -1.0 0.0;-1.0 2.0 -1.0; 0.0 -1.0 2.0')

b=np.matrix([19,45,0]).T

print('A=')
print(A)
print('b=')
print(b)

solution=np.linalg.solve(A,b)
print('numpy solution : ' + str(solution))

# Algorithm constants
NUM_STEPS=15
omega_sor=1.02
omega_ssor=1.27

# Derived algorithm compounds
D=np.matrix(np.diag(np.diag(A)))
L=np.matrix(-np.tril(A-D))
R=np.matrix(-np.triu(A-D))

M_j=D.I*(L+R)
M_gs=(D-L).I*R
M_sor=(D-(omega_sor*L)).I*((1-omega_sor)*D+omega_sor*R)
M_ssor=(D-omega_ssor*R).I*(((1-omega_ssor)*D+omega_ssor*L)*(D-omega_ssor*L).I)*((1-omega_ssor)*D+omega_ssor*R)

D_j=D.I
D_gs=(D-L).I
D_sor=omega_sor*(D-omega_sor*L).I
D_ssor=omega_ssor*(2-omega_ssor)*(D-omega_ssor*R).I*D*(D-omega_ssor*L).I

# Start vector
x0=np.matrix([0.0,0.0,0.0]).T

# Iterations
print('Jacobi:')
x_j=[]
x_j.append(x0)
for i in range(0,NUM_STEPS):
    x_j.append(M_j*x_j[-1]+D_j*b)

print(x_j[-1])

print('GS')
x_gs=[]
x_gs.append(x0)
for i in range(0,NUM_STEPS):
    x_gs.append(M_gs*x_gs[-1]+D_gs*b)

print(x_gs[-1])

print('SOR:')
x_sor=[]
x_sor.append(x0)
for i in range(0,NUM_STEPS):
    x_sor.append(M_sor*x_sor[-1]+D_sor*b)
eig_M_sor=np.linalg.eig(M_sor)
rho_M_sor=max(map(lambda x:np.linalg.norm(x), eig_M_sor[0]))
optimal_omega_sor=2/(1+np.sqrt(1-rho_M_sor*rho_M_sor))
print('Optimal omega for SOR: ' + str(optimal_omega_sor))
    
print(x_sor[-1])

print('SSOR:')
x_ssor=[]
x_ssor.append(x0)
for i in range(0,NUM_STEPS):
    x_ssor.append(M_ssor*x_ssor[-1]+D_ssor*b)
    
print(x_ssor[-1])

x_ts=[]
rho=[]
x_ts.append(x0)
rho.append(2)

print('Tschebyscheff')
eig_M_ssor=np.linalg.eig(M_ssor)
max_eig_ssor=max(eig_M_ssor[0])
min_eig_ssor=min(eig_M_ssor[0])
optimal_omega_sor=2/(2-max_eig_ssor-min_eig_ssor)
print('Optimal omega for SSOR: ' + str(optimal_omega_sor))
print('Eigenvalues in [' + str(min_eig_ssor) + ',' + str(max_eig_ssor) + ']')

gamma=2/(2-max_eig_ssor-min_eig_ssor)
F1=(2-max_eig_ssor-min_eig_ssor)/(max_eig_ssor-min_eig_ssor)
x_ts.append(gamma*(M_ssor*x_ts[-1]+D_ssor*b)+(1-gamma)*x_ts[-1])
for i in range(0,NUM_STEPS-1):
    rho.append(1/(1-1/(4*F1*F1)*rho[-1]))
    x_ts.append(rho[-1]*(gamma*(M_ssor*x_ts[-1]+D_ssor*b)+(1-gamma)*x_ts[-1])+(1-rho[-1])*x_ts[-2])

# Graph plotting
errornorm_map=lambda x:np.linalg.norm(x-solution)
errornorm_j=map(errornorm_map, x_j)
errornorm_gs=map(errornorm_map, x_gs)
errornorm_sor=map(errornorm_map, x_sor)
errornorm_ssor=map(errornorm_map, x_ssor)
errornorm_ts=map(errornorm_map, x_ts)

plt.plot(errornorm_j, label='Jacobi', ls='--', color='blue')
plt.plot(errornorm_gs, label='Gauss-Seidel', ls='-.', color='red')
plt.plot(errornorm_sor, label='SOR (omega=' + str(omega_sor) + ')', ls='-', color='green')
plt.plot(errornorm_ssor, label='SSOR (omega=' + str(omega_ssor) + ')', ls=':', color='black')
plt.plot(errornorm_ts, label='Tschebyscheff SSOR', ls='--', color='orange')
plt.ylabel('Fehler')
plt.xlabel('Iteration')
plt.legend(loc="upper right")
plt.ylim([-2,70])

# Table plotting
plt.figure()
ax=plt.gca()
col_labels=['Jacobi','GS','SOR(' + str(omega_sor) + ')','SSOR(' + str(omega_ssor) + ')','Tschebyscheff SSOR']
row_labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']

table_vals=np.array([errornorm_j,errornorm_gs,errornorm_sor,errornorm_ssor,errornorm_ts]).T

the_table = plt.table(cellText=table_vals,
                  colWidths = [0.2]*5,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center')
plt.text(-0.02, 0.04, 'Approximationsfehler',size=12)
plt.axis('off')

plt.plot()
plt.show()

