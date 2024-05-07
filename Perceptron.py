import numpy as np
class Perceptron(object):
    """ Clasificador perceptrón 
    
    Parámetros
    ----------
    eta : float
        Taza de aprendizaje (entre 0.0 y 1.0)
    n_iter : int
        Pasos sobre la muestra de entrenamiento
    random_state : int
        Semilla generadora de número aleatorio para la inicialización de pesos aleatorios
    init_weight : boolean
        Bandera para indicar la manera en que se inicializaran los pesos
    weights : array
        Lista de numeros flotantes para inicializar los pesos
    shuffle : boolean
        Bandera para indicar que en cada iteracion los valores de entrenamiento estaran ordenados de manera aleatoria
    f_activate : string
        "step" para colocar al funcion de activacion en modo paso y "sign" para modo signo    
        
    Atributos
    ---------
    weights: arreglo de 1-d
        Pesos después de entrenamiento
    errors_ : list
        Número de clasificaciones incorrectas (actualizaciones) en cada época
        
    """
    
    def __init__(self, eta = 0.01, n_iter = 10, random_state = 1, init_weight = True, weights = [], shuffle = True, f_activate = "step"):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.init_weight = init_weight
        self.shuffle = shuffle
        self.f_activate = f_activate
        self.weights = weights
        self.auto_stop = False
    
    def fit(self, X, y):
        """ Ajustar los datos de entrenamiento
        
        Parámetros
        ----------
        X : 
            Vector de entrenamiento
        y: 
            Valores objetivo
        
        Returns
        -------
        self : object
        
        """
        if self.init_weight == False:
            self.weights = np.zeros(1 + X.shape[1])
        elif not self.weights:
            rgen = np.random.RandomState(self.random_state)
            self.weights = rgen.normal(loc= 0.0, scale= 0.01, size= 1 + X.shape[1])

        self.errors_ = []
        self.old_weights = []
        if self.n_iter <= 0:
            self.n_iter = 2147483647
            self.old_weights = self.weights.copy()
            self.auto_stop = True

        for _ in range(self.n_iter):
            #implementar entrenamiento
            errors = 0
            if self.shuffle:
                data = list(zip(X,y))
                np.random.shuffle(data)
                zip(*data)
            else:
                data = zip(X,y)

            for x_i,t in data:
                update = self.eta * (t - self.predict(x_i))
                self.weights[1:] += update * x_i
                self.weights[0] += update
                errors += int(update != 0.0)
            print("Nuevos Pesos\n",self.weights)
            self.errors_.append(errors)
            if np.array_equal(self.weights,self.old_weights) and self.auto_stop:
                break
            else:
                self.old_weights = self.weights.copy()
        return self
    
    def net_input(self, X):
        """ Calcular la entrada a la red"""
        #función de entrada
        z = np.dot(X,self.weights[1:]) + self.weights[0]
        return z 
    
    def predict(self, X):
        """ Retornar la etiqueta de clase despues de la función de paso signo"""
        #función de activación paso o signo
        if self.f_activate=="step":
            phi_z = np.where(self.net_input(X) >= 0.0,1,0)
        else:
            phi_z = np.where(self.net_input(X) >= 0.0,1,-1)
        return phi_z
        
carac = np.array([[1,1],[1,0],[0,1],[0,0]])
tar = np.array([1,0,0,0])
percep = Perceptron(eta=0.5,n_iter=0,init_weight=True,shuffle=True,f_activate="step")
percep.fit(carac,tar)