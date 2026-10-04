"""
Módulo SCAN — Salvi et al. (2020)

Referencia:
    Salvi, M., Michielli, N. y Molinari, F. (2020).
    Stain color adaptive normalization (SCAN) algorithm: Separation and
    standardization of histological stains in digital pathology.
    Computer Methods and Programs in Biomedicine, 193, 105506.
    https://doi.org/10.1016/j.cmpb.2020.105506

Notas:
    - Implementación basada en el paper original.
    - La clase expone una interfaz fit() / transform() consistente
      con el resto de los algoritmos del pipeline.
    - SCAN no requiere pre-entrenamiento con el target ya que estima
      los vectores de tinción de forma adaptativa en cada imagen.
      fit() almacena los vectores del target para usarlos en transform().
"""

import numpy as np
from sklearn.cluster import KMeans


class StainNormalizerSCAN:
    """
    Normalizador de tinción basado en SCAN (Stain Color Adaptive Normalization).

    Referencia: Salvi et al. (2020).
    https://doi.org/10.1016/j.cmpb.2020.105506

    Uso
    ---
        normalizer = StainNormalizerSCAN()
        normalizer.fit(target_img)               # target: np.ndarray (H, W, 3) uint8
        norm_img = normalizer.transform(source_img)  # source: np.ndarray (H, W, 3) uint8
    """

    def __init__(self):
        self._target_stains = None

    def fit(self, target_img):
        """
        Extrae y almacena los vectores de tinción de la imagen target.

        Parámetros
        ----------
        target_img : np.ndarray uint8 de forma (H, W, 3)
        """
        self._target_stains = self._get_stains(target_img)

    def transform(self, source_img):
        """
        Normaliza la imagen fuente hacia el target.

        Parámetros
        ----------
        source_img : np.ndarray uint8 de forma (H, W, 3)

        Retorna
        -------
        np.ndarray uint8 de forma (H, W, 3) — imagen normalizada
        """
        if self._target_stains is None:
            raise AssertionError('Llamá a fit() antes de transform()')

        # Procesamiento de la imagen fuente
        od             = self._rgb_to_od(source_img)
        od_pixels, _   = self._remove_background(od)
        stain_init     = self._estimate_stain_matrix(od_pixels)
        labels         = self._structural_clustering(od_pixels, stain_init)
        stain_refined  = self._refine_stains(od_pixels, labels)

        # Matriz H de la fuente (concentraciones píxel a píxel)
        concentrations = self._compute_concentrations(od, stain_refined)

        # Reconstrucción: W_target × H_source
        return self._reconstruct_image(concentrations, self._target_stains, source_img.shape)

    # ── Métodos privados ──────────────────────────────────────────────────────

    def _rgb_to_od(self, I):
        """Convierte imagen RGB a espacio de densidad óptica via Beer-Lambert."""
        I = I.astype(np.float32) + 1
        return -np.log(I / 255.0)

    def _remove_background(self, OD, beta=0.15):
        """Elimina píxeles de fondo (blancos) usando umbral en espacio OD."""
        mask = np.any(OD > beta, axis=2)
        return OD[mask], mask

    def _estimate_stain_matrix(self, OD_pixels):
        """Estimación inicial de vectores de tinción via SVD (método Macenko)."""
        _, _, Vt = np.linalg.svd(OD_pixels, full_matrices=False)
        return Vt[:2, :]

    def _structural_clustering(self, OD_pixels, stain_matrix, n_clusters=2):
        """
        Clustering k-means de estructuras celulares.
        Separa núcleos (hematoxilina) de estroma (eosina).
        """
        projection = np.dot(OD_pixels, stain_matrix.T)
        kmeans     = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        return kmeans.fit_predict(projection)

    def _refine_stains(self, OD_pixels, labels):
        """
        Refinamiento adaptativo de vectores de tinción por estructura segmentada.
        Calcula un vector de tinción por cluster via SVD.
        """
        stains = []
        for k in np.unique(labels):
            cluster_pixels = OD_pixels[labels == k]
            _, _, Vt = np.linalg.svd(cluster_pixels, full_matrices=False)
            stains.append(Vt[0])
        return np.array(stains)

    def _compute_concentrations(self, OD, stain_matrix):
        """
        Calcula mapa de densidad H (concentraciones píxel a píxel)
        mediante mínimos cuadrados.
        """
        S           = stain_matrix.T
        OD_reshaped = OD.reshape(-1, 3).T
        return np.linalg.lstsq(S, OD_reshaped, rcond=None)[0]

    def _reconstruct_image(self, concentrations, reference_stains, shape):
        """
        Reconstruye imagen normalizada combinando H_source con W_target.
        Ecuación: V_norm = W_target × H_source → I_norm = exp(-V_norm) × 255
        """
        OD_norm = np.dot(reference_stains.T, concentrations)
        I_norm  = np.exp(-OD_norm)
        I_norm  = (I_norm * 255).clip(0, 255)
        return I_norm.T.reshape(shape).astype(np.uint8)

    def _get_stains(self, img):
        """Pipeline completo de extracción de vectores de tinción de una imagen."""
        OD           = self._rgb_to_od(img)
        OD_pixels, _ = self._remove_background(OD)
        stain_init   = self._estimate_stain_matrix(OD_pixels)
        labels       = self._structural_clustering(OD_pixels, stain_init)
        return self._refine_stains(OD_pixels, labels)
