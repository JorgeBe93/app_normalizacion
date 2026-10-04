"""
Módulo ACD Experimental — Zheng et al. (2019) — Versión Optimizada

Mejoras respecto a la versión anterior:
    - Grafo TensorFlow construido UNA SOLA VEZ en __init__()
    - Sesión TF reutilizada entre llamadas a transform()
    - Evita acumulación de nodos en memoria que causaba lentitud
    - Método close() para liberar recursos al finalizar

Parámetros explorables:
    - eta   : proporción esperada de hematoxilina (default=0.6)
    - gamma : densidad óptica media esperada (default=0.5)

Referencia:
    Zheng, Y., Jiang, Z., Zhang, H., Xie, F., Shi, J. y Xue, C. (2019).
    Adaptive color deconvolution for histological WSI normalization.
    Computer Methods and Programs in Biomedicine, 170, 107-120.
    https://doi.org/10.1016/j.cmpb.2019.01.008
"""

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

_INIT_VARPHI = np.asarray([
    [0.294, 0.110, 0.894],
    [0.750, 0.088, 0.425]
])


class StainNormalizerACDExp:
    """
    Versión experimental de ACD con eta y gamma configurables.
    Optimizada para reutilizar el grafo TF y la sesión entre llamadas.

    Uso
    ---
        normalizer = StainNormalizerACDExp(eta=0.5, gamma=0.4)
        normalizer.fit(target_img)
        norm_img = normalizer.transform(source_img)
        normalizer.close()  # liberar recursos al finalizar
    """

    def __init__(self, eta=0.6, gamma=0.5,
                 pixel_number=100000, step=300, batch_size=1500,
                 lambda_p=0.002, lambda_b=10, lambda_e=1):
        self.eta             = eta
        self.gamma           = gamma
        self._pn             = pixel_number
        self._bs             = batch_size
        self._step_per_epoch = int(pixel_number / batch_size)
        self._epoch          = int(step / self._step_per_epoch)
        self._lambda_p       = lambda_p
        self._lambda_b       = lambda_b
        self._lambda_e       = lambda_e

        self._template_dc_mat = None
        self._template_w_mat  = None

        # ── Construir el grafo TF una sola vez ────────────────────────────
        self._graph = tf.compat.v1.Graph()
        with self._graph.as_default():
            self._input_od = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, 3], name='input_od'
            )
            self._target_op, self._cd_op, self._w_op = self._build_model(
                self._input_od
            )
            self._init_op = tf.compat.v1.global_variables_initializer()

        # Sesión reutilizable
        self._sess = tf.compat.v1.Session(graph=self._graph)

    def _build_model(self, input_od):
        """Construye el modelo ACD en el grafo. Se llama solo una vez."""
        alpha = tf.compat.v1.get_variable(
            'alpha', initializer=_INIT_VARPHI[0].astype(np.float32)
        )
        beta = tf.compat.v1.get_variable(
            'beta', initializer=_INIT_VARPHI[1].astype(np.float32)
        )
        w = [
            tf.compat.v1.get_variable('w0', initializer=1.0),
            tf.compat.v1.get_variable('w1', initializer=1.0),
            tf.constant(1.0),
        ]

        sca_mat = tf.stack((
            tf.cos(alpha) * tf.sin(beta),
            tf.cos(alpha) * tf.cos(beta),
            tf.sin(alpha),
        ), axis=1)

        cd_mat  = tf.linalg.inv(sca_mat)
        s       = tf.matmul(input_od, cd_mat) * w
        h, e, b = tf.split(s, (1, 1, 1), axis=1)

        l_p1 = tf.reduce_mean(tf.square(b))
        l_p2 = tf.reduce_mean(2 * h * e / (tf.square(h) + tf.square(e)))
        l_b  = tf.square(
            (1 - self.eta) * tf.reduce_mean(h) - self.eta * tf.reduce_mean(e)
        )
        l_e  = tf.square(self.gamma - tf.reduce_mean(s))

        objective = (
            l_p1
            + self._lambda_p * l_p2
            + self._lambda_b * l_b
            + self._lambda_e * l_e
        )
        target = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=0.05
        ).minimize(objective)

        return target, cd_mat, w

    def _sampling_data(self, images):
        """Muestrea píxeles de tejido excluyendo fondo blanco."""
        pixels = np.reshape(images, (-1, 3))
        pixels = pixels[np.random.choice(
            pixels.shape[0], min(self._pn * 20, pixels.shape[0])
        )]
        od  = -np.log((np.asarray(pixels, np.float64) + 1) / 256.0)
        tmp = np.mean(od, axis=1)
        od  = od[(tmp > 0.3) & (tmp < -np.log(30 / 256))]
        od  = od[np.random.choice(od.shape[0], min(self._pn, od.shape[0]))]
        return od

    def _run_optimization(self, images):
        """Ejecuta la optimización reutilizando la sesión existente."""
        od_data = self._sampling_data(images)

        # Reinicializar variables para esta imagen
        self._sess.run(self._init_op)

        for _ in range(self._epoch):
            for step in range(self._step_per_epoch):
                self._sess.run(
                    self._target_op,
                    {self._input_od: od_data[step * self._bs:(step + 1) * self._bs]}
                )

        opt_cd = self._sess.run(self._cd_op)
        opt_w  = self._sess.run(self._w_op)
        return opt_cd, opt_w

    def fit(self, images):
        """
        Entrena el normalizador con la imagen target.
        images : np.ndarray uint8 de forma (k, H, W, 3)
        """
        opt_cd, opt_w = self._run_optimization(images)
        self._template_dc_mat = opt_cd
        self._template_w_mat  = opt_w

    def transform(self, images):
        """
        Normaliza la imagen fuente hacia el target.
        images : np.ndarray uint8 de forma (k, H, W, 3)
        Retorna np.ndarray float64 (k, H, W, 3) en [0, 255]
        """
        if self._template_dc_mat is None:
            raise AssertionError('Llamá a fit() antes de transform()')

        opt_cd, opt_w = self._run_optimization(images)
        transform_mat = np.matmul(
            opt_cd * opt_w,
            np.linalg.inv(self._template_dc_mat * self._template_w_mat)
        )
        od            = -np.log((np.asarray(images, np.float64) + 1) / 256.0)
        normed_od     = np.matmul(od, transform_mat)
        normed_images = np.exp(-normed_od) * 256 - 1
        return np.maximum(np.minimum(normed_images, 255), 0)

    def close(self):
        """Libera la sesión TF y el grafo. Llamar al finalizar el experimento."""
        if self._sess is not None:
            self._sess.close()
            self._sess = None

    def __del__(self):
        self.close()


# Alias para compatibilidad con la notebook de normalización
# Parámetros optimizados para tejido de pene: eta=0.4, gamma=0.6
class StainNormalizerACD(StainNormalizerACDExp):
    """
    Alias de StainNormalizerACDExp con parámetros optimizados
    para tejido de pene (eta=0.4, gamma=0.6).
    """
    def __init__(self, eta=0.4, gamma=0.6, **kwargs):
        super().__init__(eta=eta, gamma=gamma, **kwargs)
