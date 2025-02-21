import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2

def value2color(values,vmin=None,vmax=None,cmName='jet'):
    cmapPaths = matplotlib.colormaps.get_cmap(cmName)
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array([cmapPaths(norm(value))[:3] for value in values])
    return colors, norm

def visualize_flow(cords_org,cords_dst,img=None,colors=None,norm=None,weights=None,cmap='jet',colorbar=True,display=True,fwdVals=None):

    diff = cords_org - cords_dst
    dpi = 100
    img_height, img_width = img.shape[:2]  # Get the image dimensions
    fig_width, fig_height = img_width / dpi, img_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    if img is not None: ax.imshow(img)
    # for i in range(len(currImg_mask_coords)):
    #     ax.plot(currImg_mask_coords[i,0],currImg_mask_coords[i,1],'o',color='r')
    #     ax.plot(refNodes_mask_coords[matchInds[i],0],refNodes_mask_coords[matchInds[i],1],'o',color='b')
    if fwdVals is not None:
        # plot a diamond for negative values and a circle for positive values, size = val
        pointTypeMask = fwdVals > 0
        ax.scatter(*(cords_org[pointTypeMask].T),c=colors[pointTypeMask],s=abs(fwdVals[pointTypeMask])*40,marker='o',edgecolor='white',linewidth=0.5)
        ax.scatter(*(cords_org[~pointTypeMask].T),c=colors[~pointTypeMask],s=abs(abs(fwdVals[~pointTypeMask]))*40,marker='X',edgecolor='white',linewidth=0.5)
    if weights is not None:
        weightedSum = (weights[:,None] * diff).sum(0)
        ax.quiver(*(np.array([160,120]).T), weightedSum[0], weightedSum[1],color='black',edgecolor='white',linewidth=0.5)
    ax.quiver(*(cords_org.T), diff[:,0], diff[:,1],color=colors,edgecolor='white',linewidth=0.5)
    if colorbar: add_colobar(ax,plt,norm,cmap)
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height,0])
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if display:
        plt.show()
    else:
        # return the figure as image (same size as img imshow-ed above)
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        vis = cv2.resize(vis,(img.shape[1],img.shape[0]))
        plt.close(fig)
        return vis

def add_colobar(ax,plt,norm=None,cmap='jet'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create a ScalarMappable object with the "autumn" colormap
    if norm is None:
        norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add a colorbar to the axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")

    # Customize the colorbar
    cbar.set_label('Colorbar Label', labelpad=10)
    cbar.ax.yaxis.set_ticks_position('right')

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def drawMasksWithColors(im_,masks,colors,alpha=0.5):
    im = im_.copy() / 255.0
    viz = im.copy()#np.zeros((im.shape[0],im.shape[1],3))
    for j in range(len(masks)):
        viz[masks[j]] = colors[j]
    im =  alpha * viz + (1-alpha) * im
    return im