import numpy as np

# Función que me ayuda a calcular el f de Darcy con la ecuación de 
# Colebrook-White
def Col_Whi(D, ks, Re):
    '''Función para calcular los factores de fricción de una serie de tubos 
    usando la ecuación de Colebrook-White.
    
    Args:
        D (np.array - float): Diámetros de las tuberías
        ks (np.array - float): Rugosidades absolutas (mismas unidades que D)
        Re (np.array - float): Números de Reynolds de las tuberías
        
    Returns:
        np.array - float: Los factores de fricción de las tuberías'''
    
    # Definiendo tolerancia del solver y 
    tol = 1e-10
    err = 1e10

    # Convertir entradas a arrays (mínimo 1D) (así no hay problemas cuando la 
    # entrada es un solo número)
    D = np.atleast_1d(D)
    ks = np.atleast_1d(ks)
    Re = np.atleast_1d(Re)

    # Vector semilla de valores de f
    f0 = np.ones(len(D)) * 0.02

    # Bucle que se repite para encontrar los valores de f
    while err > tol:
    
        # Calculando los nuevos valores de f
        f1 = (-2 * np.log10(ks / D / 3.7 + 2.51 / Re / np.sqrt(f0))) ** -2

        # Estimando el valor del error como la norma del vector de diferencias
        # relativas de los factores f
        err = np.linalg.norm((f1 - f0) / f0)

        # Reemplazo los valores
        f0 = f1

    return f1

# Función para calcular una tubería simple usando la ecuación de Colebrook-White 
def TS_COLEBROOK(D, L, ks, deltaH, sumK, nu):

    """Función que calcula una tubería simple con la ecuación de Colebrook-White
    Args: 
        D: Diámetro de tubería en m
        L: Longitud de tubería en m
        ks: Rugosidad de tubería en m
        deltaH: Pérdida de carga total en m
        sumK: Suma de coeficientes de pérdidas locales
        nu: viscosidad cinemática en m2/s
        
    Returns:
        np.array: Caudal en m3/s[0], Reynolds[1], f[2]"""
    
    # Parámetros numéricos
    err = 1e10
    tol = 1e-8
    K1 = 8 / np.pi ** 2 / 9.81
    
    # Suponer un f para empezar
    f0 = 0.02

    # Iterando 
    while err > tol:

        # Estimación de caudal
        Q = np.sqrt(deltaH * D ** 4 / (K1 * (f0 * L / D + sumK)))

        # Calculando el número de Reynolds
        Re = (4 * Q) / (np.pi * D * nu)

        # Nuevo valor de f
        f1 = Col_Whi(D, ks, Re)

        # Calculando el error
        err = np.abs((f0 - f1) / f0)

        # Reemplazando valores
        f0 = f1

    # Devuelvo los valores de interés
    return Q[0], Re[0], f1[0]


# Pruebas varias para ver cómo están las funciones
# print(TS_COLEBROOK(0.293, 730, 1.5e-6, 43.5, 11.8, 1.0068e-6))
# Q = 0.3126 Re = 1349050 f = 0.01121 (lo que da a mano)

