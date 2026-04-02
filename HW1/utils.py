import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
from typing import List
import functools


def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)

    return wrap_fn


class SeamImage:
    def __init__(self, img_path: str, vis_seams: bool = True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()

        self.H, self.W = self.rgb.shape[:2] # -> original image dimaensions
        self.h, self.w = self.H, self.W

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.zeros_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        xx, yy = np.meshgrid(range(self.w), range(self.h))
        self.idx_map = np.stack((yy, xx), axis=-1)

    @NI_decor
    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        gs_img = np_img @ self.gs_weights

        # uncomment for padding (a common pracctive in image processing)
        # gs_img[0, :] = .5
        # gs_img[-1, :] = .5
        # gs_img[:, 0] = .5
        # gs_img[:, -1] = .5
        return gs_img

    @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        self.resized_gs = np.squeeze(self.resized_gs)
        
        #Using offset indices for forward differencing and padding back to the original shape
        dx = self.resized_gs[:, :-1] - self.resized_gs[:, 1:]
        dx = np.pad(dx, ((0,0), (0,1)), mode='constant')
        
        dy = self.resized_gs[:-1, :] - self.resized_gs[1:, :]
        dy = np.pad(dy, ((0,1), (0,0)), mode='constant')

        #Computing magnitude and dividing by the maximal value of sqrt(1^2+1^2) to get range [0,1]
        return np.sqrt(dx**2 + dy**2)/np.sqrt(2)


    def update_ref_mat(self):
        """ Updates matrices for seam visualization

        Guidelines & hints:
            - Given the latest computed seam, you need to track its original indices and mark them (self.cumm_mask) using self.ixd_map
            - Resize self.idx_map each seam update
        """
        #Setting each cell of the seam True in self.cumm_mask
        seam = self.seam_history[-1]
        rows = np.arange(self.h)
        coords = self.idx_map[rows, seam]
        row_orig = coords[:, 0]
        col_orig = coords[:, 1]
        self.cumm_mask[row_orig, col_orig] = True

        #Update visualization by red
        self.seams_rgb[row_orig, col_orig] = (1.0, 0, 0)

        #Removing the seam from self.idx_map using mask
        self.mask[rows, seam] = False
        self.idx_map = self.idx_map[self.mask].reshape(self.h, self.w - 1, 2)

    def reinit(self):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seams to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, mask) where:
                - E is the gradient magnitude matrix
                - mask is a boolean matrix for removed seams
            iii) find the best seam to remove and store it
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the chosen seam (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you wish, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked in red (for comparison)

        NOTE: you may not use np.gradient or other off-the-shelf tools for gradient calculation, but you can use them to compare your results.
        """
        for _ in tqdm(range(num_remove)):
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool)

            seam = self.find_minimal_seam()
            self.seam_history.append(seam)
            if self.vis_seams:
                self.update_ref_mat()
            self.remove_seam(seam)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """
        raise NotImplementedError("TODO: Implement SeamImage.find_minimal_seam in one of the subclasses")

    @NI_decor
    def remove_seam(self, seam: List[int]):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using:
        3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.

        :arg seam: The seam to remove
        """
        #Setting up a 3d mask and 1d mask for rgb and grayscale images
        rows = np.arange(self.h)
        # mask_1d = np.ones((self.h, self.w), dtype=bool)
        self.mask[rows, seam] = False
        mask_3d = np.stack([self.mask] * 3, axis=2)
        self.w -= 1

        #Update RGB and GS for the next round of seam removal
        self.resized_rgb = self.resized_rgb[mask_3d].reshape(self.h, self.w, 3)
        self.resized_gs = self.resized_gs[self.mask].reshape(self.h, self.w, 1)

    @NI_decor
    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        direction = -1 if clockwise else 1
        for name in ['rgb', 'resized_rgb', 'gs', 'resized_gs', 'idx_map']:
            mat = getattr(self, name)
            setattr(self, name, np.rot90(mat, direction))
        
        self.w, self.W, self.h, self.H = self.h, self.H, self.w, self.W

    @NI_decor
    def seams_removal_vertical(self, num_remove: int):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    @NI_decor
    def seams_removal_horizontal(self, num_remove: int):
        """ Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
        - No need to reimplement SC for horizontal seam removal!
        - Once you figure out how, this method should look like:
                SOME_OPERATION(...)
                seam_removal(...)
                SOME_OPERATION(...)
            and thats it!
        """
        self.rotate_mats(True)
        self.seams_removal(num_remove)
        self.rotate_mats(False)

    """
    BONUS SECTION
    """

    @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seams to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition")

    @NI_decor
    def seams_addition_horizontal(self, num_add: int):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_add (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    @NI_decor
    def seams_addition_vertical(self, num_add: int):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")


class GreedySeamImage(SeamImage):
    """Implementation of the Seam Carving algorithm using a greedy approach"""

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using a greedy algorithm.

        Guidelines & hints:
        The first pixel of the seam should be the pixel with the lowest cost.
        Every row chooses the next pixel based on which neighbor has the lowest cost.
        """
        #Selecting starting point on top_row based on energy + forward cost looking with 1 new edge
        seam = np.empty(self.h, dtype=int)
        
        self.resized_gs = np.squeeze(self.resized_gs)

        left_pixels = np.pad(self.resized_gs[0, :-1], (1,0), mode='edge')
        right_pixels = np.pad(self.resized_gs[0, 1:], (0,1), mode='edge')
        top_row = self.E[0] + np.abs(right_pixels - left_pixels)

        seam[0] = np.argmin(top_row)

        #For each row, considering energy + forward cost looking
        for row in range(1, self.h):
            prev_index = seam[row - 1]
            # Set up boundaries for pixel candidates, handle edges
            window = range(max(0, prev_index - 1), min(self.w, prev_index + 2))

            candidate_cost = float('inf')
            candidate = -1

            for pixel in window:
                pixel_l = self.resized_gs[row, pixel - 1] if pixel > 0 else self.resized_gs[row, 0]
                pixel_r = self.resized_gs[row, pixel + 1] if pixel < self.w - 1 else self.resized_gs[row, self.w - 1]
                pixel_up = self.resized_gs[row-1, pixel]

                cost = self.E[row, pixel] + abs(pixel_r - pixel_l)

                if (pixel == prev_index - 1): #Going ↙
                    cost += abs(pixel_r - pixel_up)

                if (pixel == prev_index + 1): #Going ↘
                    cost += abs(pixel_l - pixel_up)

                if cost < candidate_cost:
                    candidate = pixel
                    candidate_cost = cost


            seam[row] = candidate
        
        return seam


class DPSeamImage(SeamImage):
    """
    Implementation of the Seam Carving algorithm using dynamic programming (DP).
    """

    def __init__(self, *args, **kwargs):
        """ DPSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    @NI_decor
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        M = self.E.copy()

        self.resized_gs = np.squeeze(self.resized_gs)

        for row in range(1, self.h):
            for col in range(0, self.w):
                pixel_l = self.resized_gs[row, col - 1] if col > 0 else self.resized_gs[row, 0]
                pixel_r = self.resized_gs[row, col + 1] if col < self.w - 1 else self.resized_gs[row, self.w - 1]
                pixel_up = self.resized_gs[row - 1, col]

                cost_v = abs(pixel_r - pixel_l)
                cost_l = cost_v + abs(pixel_up - pixel_l)
                cost_r = cost_v + abs(pixel_up - pixel_r)

                # avoid choosing m_x if out of index to avoid rolling over
                m_l = M[row - 1, col - 1] if col > 0 else float('inf')
                m_v = M[row - 1, col]
                m_r = M[row - 1, col + 1] if col < self.w - 1 else float('inf')

                M[row, col] += min((m_l + cost_l), (m_v + cost_v), (m_r + cost_r))

        return M

    def init_mats(self):
        self.M = self.calc_M()
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, E, GS, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            E: np.ndarray (float32) of shape (h,w)
            GS: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. Changing it here may affect it on the outside.
        """
        h, w = M.shape
        for row in range(h - 1, 0, -1):
            for col in range(w):

                pixel_l = GS[row, col - 1] if col > 0 else GS[row, 0]
                pixel_r = GS[row, col + 1] if col < w - 1 else GS[row, w - 1]
                pixel_up = GS[row - 1, col]

                cost_v = abs(pixel_r - pixel_l)
                cost_l = cost_v + abs(pixel_up - pixel_l)
                cost_r = cost_v + abs(pixel_up - pixel_r)

                # avoid choosing m_x if out of index to avoid rolling over
                m_l = cost_l + M[row - 1, col - 1] if col > 0 else float('inf')
                m_v = cost_v + M[row - 1, col]
                m_r = cost_r + M[row - 1, col + 1] if col < w - 1 else float('inf')

                # by DP I.H. we are looking for previous minimum; we avoid float point comparison
                if m_l <= m_v and m_l <= m_r:
                    backtrack_mat[row, col] = col - 1
                elif m_v <= m_r:
                    backtrack_mat[row, col] = col
                else:
                    backtrack_mat[row, col] = col + 1


        return backtrack_mat
                        


    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using dynamic programming.

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (M, backtracking matrix) where:
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
        """
        self.init_mats()
        self.backtrack_mat = self.calc_bt_mat(self.M, self.E, self.resized_gs, self.backtrack_mat)

        seam = np.empty(self.h, dtype=int)
        minimal = np.argmin(self.M[self.h - 1])

        for i in range (self.h - 1, 0, -1):
            seam[i] = minimal
            minimal = self.backtrack_mat[i, minimal]

        seam[0] = minimal

        return seam

def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    raise NotImplementedError("TODO: Implement scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    raise NotImplementedError("TODO: Implement resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x, in_width, out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y, in_height, out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:, x1s] * dx + (1 - dx) * image[y1s][:, x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:, x1s] * dx + (1 - dx) * image[y2s][:, x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


