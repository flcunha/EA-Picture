import random
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import time

num_genes=500
num_vertices=6
pop_size=200
mut_chance=0.015
mut_amount=0.15
fittest_survive=True
gene_size=5+num_vertices*2
num_iter=100000
selection_cutoff=0.2
image_file='Sil2'

print(image_file + '_' + str(num_genes) + '_' + str(num_vertices) + '_' + str(pop_size))

def create_ind(parent1=None,parent2=None):
    ind=[]
    for i in range(num_genes):
        if parent1 is None or parent2 is None:
            new_gene=[random.random() for j in range(3)] #RGB
            new_gene.append(max(0.2,random.random()*random.random())) #Alpha
            new_gene.append(1) #Y/N
            x=random.random()
            y=random.random()
            for j in range(num_vertices):
                new_gene.append(x+random.random()-0.5)
                new_gene.append(y+random.random()-0.5)

        else:
            if random.random()>0.5:
                new_gene=parent1[i*gene_size:(i+1)*gene_size]
            else:
                new_gene = parent2[i * gene_size:(i + 1) * gene_size]
            for j in range(gene_size):
                if random.random()<mut_chance:
                    if j==4:
                        new_gene[j]=(new_gene[j]+1)%2
                    else:
                        new_gene[j] += random.random() * mut_amount * 2 - mut_amount;
                        if new_gene[j]>1:
                            new_gene[j]=1
                        if new_gene[j]<0:
                            new_gene[j]=0
        ind+=new_gene
    return ind

def create_pop():
    return [create_ind() for i in range(pop_size)]

# Sum up the difference between the pixel values of the reference
# image and the current individual. Subtract the ratio of this
# difference and the largest possible difference from 1 in order
# to get the fitness.

def make_image(ind,final=False):
    if final:
        use_height=height*5
        use_width=width*5
    else:
        use_height=height
        use_width=width

    output = np.zeros((use_height, use_width, 3), np.uint8)
    overlay = np.zeros((use_height, use_width, 3), np.uint8)
    for i in range(num_genes):
        if ind[i*gene_size+4]:
            #color = ind[i*gene_size:i*gene_size+3]
            color=[i*255 for i in ind[i*gene_size:i*gene_size+3]]
            alpha = ind[i*gene_size+3]

            coordinates = []
            for j in range(num_vertices):
                x, y = ind[i*gene_size+5+j*2:i*gene_size+7+j*2]
                coordinates += [([int(x * use_width), int(y * use_height)])]  # list of coordinates
                # draw the polygon on to the overlay image

            global timefillpoly
            start=time.time()
            cv2.fillPoly(overlay, np.array([np.array(coordinates)]), color)
            timefillpoly+=time.time()-start

            global timeaddWeighted
            start=time.time()
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            timeaddWeighted+=time.time()-start

            overlay = output.copy()
    return output

def calc_fit(ind):
    start=time.time()
    ind_image=make_image(ind)
    global time1
    time1+=time.time()-start

    start=time.time()
    diff=image-ind_image
    global time2
    time2+=time.time()-start

    start=time.time()

    #FLC: this next is taking a lot of time, which is a problem

    a = image - ind_image
    b = np.uint8(image < ind_image) * 254 + 1
    diff = a * b
    diff = np.ravel(diff)
    #diff = np.sum(diff)
    #return (100.0 - (diff / (width*height*3*255) * 100.0))
    diff = np.sum(np.uint64(diff) ** 2)
    return (100.0 - (diff / (width*height*3*255*255) * 100.0))

    #fit=100*(1-np.sum(diff.astype('uint64') ** 2)/(max_diff))
    #global time3
    #time3+=time.time()-start
    #return fit

def calc_fit_pop(pop):
    return [calc_fit(ind) for ind in pop]

def sort_pop(pop,pop_fit):
    zipped_lists = zip(pop_fit, pop)
    sorted_zipped_lists = sorted(zipped_lists,reverse=True)
    pop = [element for _, element in sorted_zipped_lists]
    pop_fit = [element for element,_ in sorted_zipped_lists]
    return pop,pop_fit

def iterate_pop(pop):
    new_pop=[]
    selectCount = math.floor(pop_size * selection_cutoff)
    randCount = math.ceil(1 / selection_cutoff)

    if fittest_survive:
        randCount-=1
        new_pop=pop[:selectCount]

    for i in range(selectCount):
        for j in range(randCount):
            randIndividual = i

            while (randIndividual == i):
                randIndividual = random.randrange(selectCount*2)
            new_ind=create_ind(pop[i],pop[randIndividual])
            new_pop.append(new_ind)

    new_pop_fit=calc_fit_pop(new_pop)
    return sort_pop(new_pop, new_pop_fit)


a=1
image=cv2.imread(image_file + '.jpg')
height, width, _ =image.shape
max_diff=width*height*3*255*255
img_blue,img_green,img_red=cv2.split(image)
image=cv2.merge((img_red, img_green, img_blue))

pop=create_pop()

time1=0
time2=0
time3=0
timeaddWeighted=0
timefillpoly=0
pop_fit=calc_fit_pop(pop)
#print(time1)
#print(time2)
#print(time3)

pop,pop_fit=sort_pop(pop,pop_fit)


for iter in range(num_iter):
    start = time.time()
    pop,pop_fit=iterate_pop(pop)
    #print("iterate_pop: " + str(time.time() - start))
    if iter%10==0:
        print("Iter: " + str(iter) + ", max fit: " + str(pop_fit[0]))
    if iter%100==0:
        curr_image=make_image(pop[0],True)
        #cv2.imwrite('curr_image.jpg',curr_image)
        plt.imshow(curr_image)
        plt.axis('off')
        plt.savefig(image_file + '_' + str(num_genes) + '_' + str(num_vertices) + '_' + str(pop_size) + '.jpg')
        plt.clf()
        #plt.show()
