import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

def execute_model(A, B, model):
    SIDE_LENGTH = 90
    FEATURE_LENGTH = 128

    inner_model = model.layers[2]

    F_A = tf.reshape(inner_model(A)[:10], (10, SIDE_LENGTH ** 2, FEATURE_LENGTH))
    F_B = tf.reshape(inner_model(B)[:10], (10, SIDE_LENGTH ** 2, FEATURE_LENGTH))
    cc = tf.linalg.matmul(F_A, F_B, transpose_b=True)
    cc = tf.nn.softmax(cc, axis=-1)
    cc = tf.reshape(cc, [10] + [SIDE_LENGTH] * 4)
    cc = np.array(cc)



    import scipy.ndimage
    grid = np.array(
      [
        [
            [
                scipy.ndimage.measurements.center_of_mass(cc[k, i, j].transpose())
                for i in range(SIDE_LENGTH)
            ]
            for j in range(SIDE_LENGTH)
        ]
       for k in range(10)]
    )
    return cc, grid

def maybe_savefig(prefix, name):
    if prefix is not None:
        import footsteps
        fname = os.path.join(footsteps.output_dir, prefix, name)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname)

def draw_grid(grid):
    plt.plot(grid[0, :, :, 0], grid[0, :, :, 1])
    plt.plot(grid[0, :, :, 0].transpose(), grid[0, :, :, 1].transpose())
    plt.ylim(SIDE_LENGTH, 0)

    plt.scatter(grid[0, :, :, 0], grid[0, :, :, 1])

    plt.ylim(SIDE_LENGTH, 0)
    plt.show()
def visualize_ten_displacements(A, B, model, prefix=None):
    
    SIDE_LENGTH = A.shape[1] - 30
    cc, grid = execute_model(A, B, model)

    #grid[:, :, 0] = scipy.ndimage.gaussian_filter(grid[:, :, 0], 1)
    #grid[:, :, 1] = scipy.ndimage.gaussian_filter(grid[:, :, 1], 1)
    frames_across = 4
    plt.figure(figsize=(19, 37))
    for k in range(10):
        row_ofs = k * frames_across
        
        plt.subplot(10, frames_across, 1 + row_ofs)
        plt.imshow(A[k, 15:-15, 15:-15, 0])
        plt.subplot(10, frames_across, 2+ row_ofs)
        plt.imshow(B[k, 15:-15, 15:-15, 0])
        plt.subplot(10, frames_across, 3+ row_ofs)
        plt.ylim(SIDE_LENGTH, 0)
        plt.imshow(B[k, 15:-15, 15:-15, 0] * 0)
        plt.scatter(grid[k, :, :, 0], grid[k, :, :, 1], c=np.array(A)[k, 15:-15, 15:-15, 0].transpose(), s=.5)
        plt.subplot(10, frames_across, 4+ row_ofs)
        g = grid[k].transpose(1, 0, 2)
        plt.imshow(g[:, :, 0] - np.expand_dims(np.arange(90), 0))
        plt.colorbar()
    maybe_savefig(prefix, "figures.png")

def visualize_featuremaps(F_A, F_B):
    f_a = np.array(F_A)#.reshape(90, 90, 128)
    f_b = np.array(F_B)#.reshape(90, 90, 128)


    # In[30]:


    f_a = f_a.reshape(-1, 128)
    f_b = f_b.reshape(-1, 128)


    # In[31]:



    from sklearn import decomposition
    pca = decomposition.PCA(n_components=30)

    x_a = pca.fit_transform(f_a)
    x_b = pca.fit_transform(f_b)


    # In[32]:


    x.shape


    # In[33]:


    x_a = x_a.reshape(10, 90, 90, 30)
    x_b = x_b.reshape(10, 90, 90, 30)


    # In[34]:


    dd = 0
    dd += 1
    dd %= 10
    plt.imshow(B[dd])
    plt.show()
    plt.imshow(f_a.reshape(10, 90, 90, 128)[dd, :, :, 0:3] / 2)
    plt.show()
    plt.imshow(f_b.reshape(10, 90, 90, 128)[dd, :, :, 0:3] / 8)


    # In[35]:


    dd = 0
    #dd += 1
    #dd %= 10
    plt.imshow(B[dd])
    plt.show()
    plt.imshow(np.sum(f_a.reshape(10, 90, 90, 128), axis=-1)[dd])
    plt.show()
    plt.imshow(f_b.reshape(10, 90, 90, 128))


    # In[84]:


    plt.plot(pca.components_[0])


    # In[162]:


    plt.plot(pca.components_[0])


    # In[44]:

def visualize_weights(model):
    w = model.get_weights()


    # In[45]:


    plt.figure(figsize=(5, 23))
    for i in range(3):
        for j in range(10):
            plt.subplot(10, 3, 3 * j + i + 1)
            plt.imshow(w[8][:, :, i, j + 10])


    # In[79]:


    model.save("longtrain_longgap")
    model.save_weights("longtrain_longgap_w")


    # In[39]:


    gen.gi_frame.f_locals

def do_nothing_model(initial_frame, initial_mask, prev_frame, prev_mask, current_frame):
    return initial_mask

from tensorflow_.models import make_model

class FMAPICON_model:
    def __init__(weights_path):
        self.inner_model = make_model(.6, 90, 128)
        self.inner_model.load_weights(weights_path)
    def __call__(self, initial_frame, initial_mask, prev_frame, prev_mask, current_frame):
        A = initial_frame[None, ::4, 215:-215:4]    
        B = current_frame[None, ::4, 215:-215:4]
        
        cc, grid = execute_model(A, B, model)

        channels = np.unique(initial_mask)

        print(grid.shape)
        warped_mask = initial_mask[::4, ::215:-215:4]

        return initial_mask
