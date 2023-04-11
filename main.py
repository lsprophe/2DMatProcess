import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


class Graph:
    def __init__(self, img):
        self.img = img
        self.visited = np.zeros_like(img, np.bool_)
        self.groups = []
        self.group_curr = -1

        self.sz = img.shape
    
    def visit(self, idx):
        # update visited flag
        self.visited[idx] = True
        # add to current wrinkle group
        self.groups[self.group_curr].append(idx)

    def neighbours(self, idx, iftrue=True):
        # find neighbouring points indices
        r = idx[0]
        c = idx[1]
        # list of indices of neighbours
        rn = range(max(r-1, 0),min(r+2, self.sz[0]))
        cn = range(max(c-1, 0),min(c+2, self.sz[1]))
        n_idx = []
        for i in rn:
            for j in cn:
                if not(i==r and j==c):
                    n_idx.append((i,j))

        # only return indices where img is true
        if iftrue:
            n_idx = [i for i in n_idx if self.img[i]]
        return n_idx
    
    def new_group(self):
        self.groups.append([])
        self.group_curr += 1

def main():
    foldername = 'C:/Users/laurp/Code/Projects/FlakeFinder/wrinkledetect/dataset/ready/'

    #filename_base = 'np2_3-n117-spincoat-0etoh-16gray'
    #filename_base = 'np2_3-spincoat-n117-25etoh-fdown-l2-16gray'
    #filename_base = 'np2_3-spincoat-n117-50etoh-fup-16gray'
    filename_base = 'np2_3-spincoat-n117-75etoh-16gray'
    
    f_img_filepath = foldername + filename_base + '-flakes.png'
    img_filepath = foldername + filename_base + '.png'

    img = read_image(img_filepath)
    f_img = read_image(f_img_filepath)
    outimg = './outputs/' + filename_base + 'wrinkles.png'

    
    # # 0% EtOH sample
    # vscale = 54.7
    # hscale = 3966.34

    # # 25% EtOH sample
    # vscale = 64.4
    # hscale = 4368.39

    # # 50% EtOH sample
    # vscale = 80
    # hscale = 5000

    # 75% EtOH sample
    vscale = 86
    hscale = 4782.03
  
    # #100% EtOH sample 1
    # vscale = 
    # hscale = 5000

    # #100% EtOH sample 2 
    # vscale =
    # hscale = 5000

    peaks, valleys = find_wrinkles(img, flake_img=f_img, out_img=outimg, plot_on=True)
    n_peaks, n_valleys, heights_av, heights_mid, widths = calculate_wrinkle_values(img, peaks, valleys, vscale, hscale, out_img=outimg)
    SA_graphene = calculate_coverage(hscale, f_img)

    w_inds = range(n_peaks)
    n_px_peak = [len(p) for p in peaks]
    csv_rows = zip(w_inds, heights_av, heights_mid, widths)
    # write to csv

    csv_name = './outputs/' + filename_base + '.csv'
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', lineterminator='\n')
        writer.writerow(["number of peaks:", n_peaks])
        writer.writerow(["number of valleys:", n_valleys])
        writer.writerow(["graphene surface area:", SA_graphene])
        writer.writerow([])
        writer.writerow(["wrinkle ind", "average height", "middle height", "middle width"])
        writer.writerows(csv_rows)

    heights_av_hist = plt.hist(heights_av)
    plt.title("Wrinkle Heights, NP Graphene LB")
    plt.xlabel("Wrinkle Average Heights (nm)")
    plt.ylabel("Count")
    plt.show()

    heights_mid_hist = plt.hist(heights_mid)
    plt.title("Wrinkle Heights, NP Graphene LB")
    plt.xlabel("Wrinkle Middle Heights (nm)")
    plt.ylabel("Count")
    plt.show()

    widths_hist = plt.hist(widths)
    plt.title("Wrinkle Widths, NP Graphene LB")
    plt.xlabel("Wrinkle Widths (nm)")
    plt.ylabel("Count")
    plt.show()

def read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    assert img is not None
    return img

def calculate_coverage(hscale, mask):
    if mask.any():
        mask_bool = mask.astype(np.bool_)
        percent_coverage = np.sum(mask_bool) / mask_bool.size
        print(percent_coverage)
    else:
        percent_coverage = 1
    total_area = hscale**2
    area_coverage = percent_coverage*total_area
    return area_coverage

def DFS(graph, idx):
    graph.visit(idx)
    nidxs = graph.neighbours(idx)
    for n in nidxs:
        if not graph.visited[n]:
            DFS(graph, n)

def euclid_dist(px1, px2):
    return np.sqrt((px1[0]-px2[0])**2 + (px1[1]-px2[1])**2)

def find_closest(px, group):
    c_dist = np.inf
    c_ind = np.empty((1,2))
    for i in group:
        dist = euclid_dist(px, i)
        if dist < c_dist:
            c_dist = dist
            c_ind = i
    
    return c_ind, c_dist

def geo_avg(w):
    avg_x = int(np.round(sum(n[1] for n in w) / len(w)))
    avg_y = int(np.round(sum(n[0] for n in w) / len(w)))
    return avg_x, avg_y

def filter_wrinkles(peaks, valleys):
    for i, p in enumerate(peaks):
        std_x = np.std(px[0] for px in p)
        std_y = np.std(px[1] for px in p)

        #if (std_x/len(p))<

def calculate_wrinkle_values(img_gray, peaks, valleys, vscale, hscale, plot_on=True, out_img=None):
    hscale_px = hscale / (img_gray.shape[0])
    # find pixel closest to geometric middle of peak
    heights_avg = np.empty(len(peaks))
    heights_mid = np.empty(len(peaks))
    widths = np.empty(len(peaks))

    # find valley closest to geometic middle of peak
    val_idxs = []

    i = 0
    for i, w in enumerate(peaks):
        avg_x_p, avg_y_p = geo_avg(w)

        mpx_p, mdist = find_closest((avg_x_p, avg_y_p), w)

        # calculate horizontal distsnace
        temps = []

        for v in valleys:
            temp, temp_dist = find_closest(mpx_p, v)
            temps.append(temp)

        vpx, dist = find_closest(mpx_p, temps)
        val_idx = temps.index(vpx)

        val = valleys[val_idx]
        val_idxs.append(val_idx)

        avg_x_v, avg_y_v = geo_avg(val)
        mpx_v, mdist_v = find_closest((avg_x_v, avg_y_v), val)
        
        hdist = euclid_dist(mpx_v, mpx_p)
        wid = hdist * hscale_px

        # NOTE: unclear how I should do the height??
        avg_peak = np.mean(img_gray[w])
        avg_val = np.mean(img_gray[val])

        h_av = (avg_peak - avg_val)/65535 * vscale

        h_mid = (img_gray[mpx_p] - img_gray[mpx_v])/65535 * vscale

        heights_avg[i] = h_av
        heights_mid[i] = h_mid
        widths[i] = wid

    return len(peaks), len(valleys), heights_avg, heights_mid, widths        


def group_wrinkles(w_img, boundaries=None):
    # graph traversal to find all points that are connected
    w_img = w_img.astype(np.uint16)
    if boundaries is not None:
        boundaries = boundaries.astype(np.uint8)
        masked_img = cv2.bitwise_and(w_img, w_img, mask=boundaries)
    else:
        masked_img = w_img
    graph = Graph(masked_img)
    it = np.nditer(masked_img, flags=['multi_index'])

    for i in it:
        if i and (not graph.visited[it.multi_index]):
            graph.new_group()
            DFS(graph, it.multi_index)
    
    # done depth first search, return wrinkle groups
    return graph.groups

def find_wrinkles_by_flake_size(img, flake_imgs, v_scale, h_scale, nn=3, plot_on=False):
    # get list of wrinkles in every flake:
    f_peaks = []
    f_valleys = []

    for f_img in flake_imgs:
        peaks, valleys = find_wrinkles(img, flake_img=f_img)
        f_peaks.append(peaks)
        f_valleys.append(valleys)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find surface area of each sheet when unwrinkled
    for f_img in flake_imgs:
        # count number of points for delaunay triangulation
        n_pts = np.sum(f_img)
        # make points for delaunay factorization
        points = np.empty(n_pts,3)
        
        # iterate over indices in flake region
        it = np.nditer(f_img, flags=['multi_index'])

        n = 0
        for i in it:
            # if the region is in a flake, add its x y z coords to the points array
            if i:
                x = i[1] * h_scale
                y = i[0] * h_scale
                z = (img_gray[it] / 255) * v_scale

                points[n, :] = x,y,z
                n += 1
    
    tri = Delaunay(points[:,:2]) # triangulate projections
    ax = plt.figure().gca(projection='3d')
    ax.plot_trisurf(
        points[:,0], points[:,1], points[:,2],
        triangles=tri.simplices
    )
    plt.show()
    
def find_wrinkles(img, flake_img=None, out_img=None, nn=3, buff=0, plot_on=False):
    img_colour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # find pixels where some number of the surrounding pixels are of different sign
    w_img = np.empty_like(img)
    sz0 = img.shape[0]
    sz1 = img.shape[1]
    it = np.nditer(img, flags=['multi_index'])
    pos = lambda x: x > 0

    # gaussian blur image to remove 'fake' wrinkles
    #img = cv2.blur(img, (5,5))
    for i in it:
        # locate how many opposite sign pixels surround the element
        # iterate through neighbours
        r = it.multi_index[0]
        c = it.multi_index[1]
        # check how many of its neighbous it is greater than
        n = img[max(r-nn, 0):min(r+nn, sz0), max(c-nn, 0):min(c+nn, sz1)]
        num = np.sum((n + buff) < i)
        w_img[it.multi_index] = num

    hills = w_img > 24
    valleys = w_img < 11

    wrinkle_peaks = group_wrinkles(hills, boundaries=flake_img)
    wrinkle_valleys = group_wrinkles(valleys, boundaries=flake_img)

    # remove wrinkles that are less than 15 pixels
    wrinkle_peaks = [xi for xi in wrinkle_peaks if len(xi) > 30]
    wrinkle_valleys = [xi for xi in wrinkle_valleys if len(xi) > 30]

    if plot_on or out_img:
        # show on image where identified wrinkles are
        for w in wrinkle_peaks:
            for i in w:
                img_colour[i] = [0,65535,0]
        for w in wrinkle_valleys:
            for i in w:
                img_colour[i] = [65535,0,0]

        if out_img:
            cv2.imwrite(out_img, img_colour)

        if plot_on:
            cv2.imshow('wrinkles', img_colour)
            cv2.waitKey()

    return wrinkle_peaks, wrinkle_valleys

if __name__ =='__main__':
    main()
