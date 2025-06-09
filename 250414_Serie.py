#  =============================================================================
# Problema para calcular el caudal total que pasa por un sistema de n tuberías
# en serie con la ecaución de Darcy-Weisbach
# Antonio Preziosi Ribero
# Abril de 2025
#  =============================================================================

# Importo las librerías necesarias para resolver el problema
import numpy as np
import pandas as pd 
from Funciones import Col_Whi


# VALORES DE ENTRADA DEL PROBLEMA (ejemplo libro Saldarriaga, ed3, Cap5)
# Se le pone un punto a algunas unidades para que Python las interprete como 
# números con decimales. 

# Pérdida total en m
hT =  28.5

# Longitudes de tuberías en m
L = np.array([423, 174, 373., 121])

# Diámetros de tuberías en mm
D = np.array([600, 500, 300., 250])

# Rugosidades en mm
ks = np.array([0.3, 0.0015, 0.0015, 0.0015])

# Suma de coeficientes de pérdidas menores
Sum_K = np.array([4.2, 3.4, 5.3, 7.5])

# Caudales de derivación en L/s
ql = np.array([60, 74, 60.])

# Viscosidad cinemática de fluido m2/s
nu = 1.14e-6

# Transformación a unidades consistentes (de mm a m y de L/s a m3/s)
D /= 1000.
ks /= 1000.
ql /= 1000.

# Parámetros numéricos para detener el solver
err = 1e10
tol = 1e-8
K1 = 8 / np.pi ** 2 / 9.81  # Constante carga de velocidad 0.0826Q^2/D^4

# Paso 1 del problema, se suponen los f de las tuberías (por simplicidad, se 
# usará 0.02 para todas las tuberías. Luego, el algoritmo irá corrigiendo)
f0 = np.ones(len(D)) * 0.015

# La suma de los ql es solamente una suma acumulada de los caudales que van 
# saliendo del sistema
sql = np.cumsum(ql)
sql = np.append(0, sql)

it = 1

# Iterando en ciclo while para obtener el resultado
while err > tol:

    # Sacando los valores del vector alpha
    alpha = (1 / D ** 4) * (f0 * L / D + Sum_K)

    # Valores para sumatorias compuestas
    alpha_q = alpha * sql
    alpha_q2 = alpha * sql ** 2

    # Valores de la cuadrática
    A = sum(alpha)
    B = -2 * sum(alpha_q)
    C = sum(alpha_q2) - hT / K1

    # Solución de la cuadrática
    QT = np.roots([A, B, C])

    # Me quedo con la positiva
    QT = QT[QT >= 0]

    # Calculo los caudales de cada tubo
    Qi = QT - sql

    # Calculo los Reynolds para cada tubo
    Re = 4 * Qi / (np.pi * D * nu)

    # Con los Reynolds estimo los nuevos valores de f
    f1 = Col_Whi(D, ks, Re)
    
    # Calculo el error como la norma de la diferencia de f1 y f0
    err = np.linalg.norm((f1 - f0) / f0)

    # Reemplazo valores y sigo
    f0 = f1

    # Imprimo los resultados intemedios para verificar y revisar la convergencia
    # Incluye la impresión del error relativo en el vector de las f calculadas
    # vs las f supuestas
    print('\n\nIteración: %.i \tError: %.2e'%(it, err))
    
    print(pd.DataFrame({
        'Q (L/s)': Qi * 1000,
        'V (m/s)': Qi * 4 / np.pi / D ** 2,
        'Re': Re,
        'f': f1,
        'hf (m)': f1 * L / D ** 5 * K1 * Qi ** 2,
        'hk (m)': Sum_K * K1 * Qi ** 2 / D ** 4
    }))
    it += 1

# En este dataframe se presentan los resultados de forma ordenada para poder
# hacer las revisiones contra el texto
resultados = pd.DataFrame({
              'Diam. (mm)': D * 1000,
              'Long (m)': L,
              'Rug. (mm)': ks * 1000,
              'Suma acc.': Sum_K,
              'Caudal (L/s)': Qi * 1000,
              'Vel. (m/s)': 4 * Qi / (np.pi * D ** 2), 
              'he (m)': (f1 * L / D + Sum_K) * K1 * Qi ** 2 / D ** 4,
              'hf (m)': f1 * L / D ** 5 * K1 * Qi ** 2,
              'hk (m)': Sum_K * K1 * Qi ** 2 / D ** 4,
              'f': f1, 
              'Re': Re})

resultados.style.format({
              'Diam. (mm)': '{:.0f}',
              'Long (m)': '{:.0f}',
              'Rug. (mm)': '{:.4f}',
              'Suma acc.': '{:.1f}',
              'Caudal (L/s)': '{:.2f}',
              'Vel. (m/s)': '{:.3f}',
              'he (m)': '{:.2f}', 
              'hf (m)': '{:.2f}',
              'hk (m)': '{:.2f}',
              'f': '{:.5f}', 
              'Re': '{:.1f}'
})

print('\n' * 8)
print(resultados)

print('\n' * 3)
print('Pérdidas por fricción totales: %.3f'%(sum(resultados['hf (m)'])))
print('Pérdidas por accesorios totales: %.3f'%(sum(resultados['hk (m)'])))
print('La pérdida total calculada es de: %.3f m'%(sum(resultados['he (m)'])))
print('El caudal que transcurre por el sistema antes de las derivaciones es ' + 
      'de: %.2f L/s'%(resultados['Caudal (L/s)'][0]))