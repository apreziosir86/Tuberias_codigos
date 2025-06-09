import numpy as np
import pandas as pd
from Funciones import Col_Whi, TS_COLEBROOK

# DATOS FÍSICOS DEL PROBLEMA (LO QUE SE PUEDE MEDIR Y QUE HACE PARTE DEL 
# PROBLEMA)

# Diámetros de los tubos en mm y los paso a m
D = np.array([450, 300, 300.])
D /= 1000

# Longitudes de la tubería en m
L = np.array([278, 312, 312])

# Rugosidades en mm y las paso a m
ks = np.ones(len(L)) * 0.046
ks /= 1000

# Coeficientes de pérdidas por accesorios
Sum_K = np.array([7.7, 9.5, 9.5])

# Densidad del fluido en kg/m3
rho = 860

# Viscosidad dinámica del fluido en Pa s
mu = 7.19e-3
nu = mu / rho            # Calculando la viscosidad cinemática

# Caudal total que trae el sistema en L/s y lo paso a m3/s
Qt = 460.
Qt /= 1000

# Presión antes de la bifurcación en kPa
PA = 875

# PARÁMETROS NUMÉRICOS DE LA SOLUCIÓN QUE SE QUIERE PLANTEAR
err = 1e10                       # Error inicial (valor muy alto)
tol = 1e-8                       # Tolerancia del solver (valor chiquito)
f0 = np.ones(len(D)) * 0.01      # Suposición inicial de valores de f
Qi = np.zeros(len(D))            # Vector que almacena caudales
Re = np.zeros_like(Qi)           # Vector que almacena numeros de Reynolds
f1 = np.zeros_like(Qi)           # Vector para nuevos valores de f
K1 = 8 / np.pi ** 2 / 9.81       # Constante de carga de velocidad
it = 1                           # Contador de iteraciones

# Empieza el bucle de iteraciones
while err >= tol:

    # Calculando los ki con los valores de las f supuestas
    raizCi = np.sqrt(f0 * L / D + Sum_K)

    # Calculo la suma de D2 / raizci y la elevo al cuadrado
    factorp = sum(D ** 2 / raizCi) ** -2

    # Con ese factor hallo la pérdida de carga en m
    deltaH = factorp * K1 * Qt ** 2

    # Calculo las tuberías simples con esa pérdida para obtener el caudal, 
    # número de Reynolds y el nuevo factor de fricción
    for i, (diam, long, rug, sumK) in enumerate(zip(D, L, ks, Sum_K)):

        # Calculando el caudal, Reynolds y factor f para cada tubería. Los D 
        # entran en m (como vienen), las long en m, las rugo en m, el deltaH en 
        # m. La suma de los K es adimensional y el nu en unidades consistentes
        Qi[i], Re[i], f1[i] = TS_COLEBROOK(diam, long, rug, deltaH, sumK, nu)

    # Ajusto los caudales para que cumplan con conservación de masa
    Qi *= Qt / sum(Qi)

    # Con los caudales recalculados vuelvo a hacer Reynolds y f (con esto 
    # garantizo la conservación de masa). Estoy reescribiendo, pero no hay 
    # problema.
    Re = 4 * Qi / (np.pi * D * nu)
    f1 = Col_Whi(D, ks, Re)        

    # Con los f calculados estimo el error de la aproximación
    err = np.linalg.norm((f1 - f0) / f0)
    
    # Imprimiendo los resultados parciales en la pantalla
    print('\n\nIteracion: %.i \tError: %.3e \tCaudal total: %.2f L/s'%(it, err,
                                            sum(Qi) * 1000))
    print(pd.DataFrame({
        'Q (m3/s)': Qi,
        'V (m/s)': 4 * Qi / (np.pi * D ** 2),
        'ht (m)': (f1 * L / D + Sum_K) * K1 * Qi ** 2 / D ** 4,
        'hf (m)': (f1 * L / D) * K1 * Qi ** 2 / D ** 4,
        'hk (m)': Sum_K * K1 * Qi ** 2 / D ** 4,
        'Re': Re,
        'f': f1
    }))

    # Reemplazo los valores y sigo
    f0 = f1
    it += 1