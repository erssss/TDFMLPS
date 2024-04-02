import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from queue import PriorityQueue

class graph:
    def __init__(self,multi_y,delta_y,method="dumb",cent=True,idx=0):
        self.multi_y=multi_y
        self.delta_y=delta_y
        self.n = self.multi_y.shape[0]-1
        self.m = self.multi_y.shape[1]
        self.method = method
        self.cent = cent
        if method == "mid": # don't use it
            self.goal = (self.n,list(multi_y[self.n,:]).index(sorted(multi_y[self.n,:])[int(self.n/2)]))
            self.start = (0,list(multi_y[0,:]).index(sorted(multi_y[0,:])[int(self.n/2)]))
        elif method == "dumb":
            self.goal = (self.n,idx)
            self.start = (0,idx)
        self.max_dy = np.max(self.multi_y)-np.min(self.multi_y)

    def getGoal(self):
        return self.goal

    def getStart(self):
        return self.start

    def cost(self,current, next):
        if current[0]==-1 or next[0] == self.n+1:
            return 0
        dis = abs(self.multi_y[current[0]][current[1]]-self.multi_y[next[0]][next[1]])
        cent = self.delta_y[next[0]][next[1]]
        return dis+cent

    def neighbors(self,current):
        if current[0]+1 == self.multi_y.shape[0]-1:
            return [self.goal]
        else:
            return [(current[0]+1,i) for i in range(self.multi_y.shape[1])]

    def heuristic(self,goal, next):
        x = next[0]
        if x == self.n+1:
            return 0
        return abs(goal[1]-self.multi_y[x][next[1]])
    
def cal_all_regloss(multi_y,X,n,m,k):
    regloss = np.zeros((n-k+1,pow(m,k)))
    for i in range(n-k+1):
        for j in range(pow(m,k)):
            sel = []
            index = j
            for p in range(k):
                index = (j %pow(m,k-p))//pow(m,k-p-1)
                sel.append(multi_y[i+p][index])
            # print(i,j,sel)
            regloss[i][j] = get_k_reg(X[i:i+k],sel)
    return regloss

def DFRT(multi_y,X,win):
    mid = np.median(multi_y,axis=1)
    delta_y = abs(multi_y-mid.reshape(-1,1))
    n = X.shape[0]
    num = int(n/win)
    selection = []
    last = 0
    intervals = np.linspace(0, n, num+1, dtype=int)
    # print(num,intervals)
    for i in intervals[1:]:
        selection.extend(A_star(i-last, multi_y[last:i], delta_y[last:i], "dumb"))
        last = i
    return selection

def dfs(multi_y,X,points,k,apploss,regloss):
    n=multi_y.shape[0]
    m=multi_y.shape[1]
    path_loss = [0 for i in range(n)]
    path=[[-1,0]]
    depth = 0
    cost = 0
    best_sel = []
    min_cost = apploss
    while(path!=[]):
        path[-1][0]+=1
        depth = len(path)
        if path[-1][0]>=m:
            while( path[-1][0]==m):
                path.pop()
                depth-=1
                if len(path)>0:
                    path[-1][0]+=1
                else:
                    break
        if depth==0:
            break
        point=points[depth-1][path[-1][0]]
        if depth>=k:
            index = sum([path[depth-k+i][0]*pow(m,k-i-1) for i in range(k)])
            loss_ = regloss[depth-k,index]

            path[-1][1]=loss_
            cost =sum([path[i][1] for i in range(len(path))])
            if cost>min_cost:
                continue
            if depth==n:
                min_cost=min(min_cost,cost)
                if cost == min_cost:
                    best_sel=copy.deepcopy(path)
                continue
        path.append([-1,0])
    return best_sel,min_cost

def bfs(multi_y,X,k):
    n=multi_y.shape[0]
    m=multi_y.shape[1]
    sels,paths_loss = bfs_p(multi_y[0:k,:],X[0:k],k,edge=-1)
    paths = sels
    for i in range(1,n-k):
        sels,paths_loss = bfs_p(multi_y[i:i+k,:],X[i:i+k],k,edge=0,before_loss=paths_loss)
        paths = np.hstack((paths,sels))
    sels,paths_loss = bfs_p(multi_y[n-k:n,:],X[n-k:n],k,edge=1,before_loss=paths_loss)
    paths = np.hstack((paths,sels))
    # print("paths",paths,"path_loss",paths_loss)
    return paths,paths_loss

def a_star(graph,no_h=False):
    frontier = PriorityQueue()
    goal = graph.getGoal()
    start = graph.getStart()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    new_cost = 0
    while not frontier.empty():
        current = frontier.get()
        if current[0] == goal[0]:
           break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                if no_h:
                    priority = new_cost
                else:
                    priority = new_cost + graph.heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    return came_from,goal,start,cost_so_far[graph.goal]


def regression(sel,X,p):
    n = X.shape[0]
    w = (np.sum(X*sel)-(1/n)*np.sum(X)*np.sum(sel)) / (np.sum(X**2) - (1/n) * (np.sum(X))**2)
    b = np.mean(sel)-w*np.mean(X)
    y_pred = (p*w+b).flatten()

    return y_pred


def A_star(n,multi_y,delta_y,method="dumb",cent=False,no_h=False,edge=0):
    m = multi_y.shape[1]
    path = []
    start = 0
    end = 0
    cost = sys.float_info.max
    for idx in range(0,m):
        g = graph(multi_y,delta_y,method,cent,idx=idx)
        path_,end_,start_,new_cost = a_star(g,no_h) 
        if new_cost<cost:
            path = path_
            start = start_
            end = end_
        # print(idx,new_cost)
    p = end
    selection = []
    while p!=start:
        selection.append(p[1])
        p = path[p]
    selection.append(start[1])
    # print(selection)
    selection_y = []
    for i in range(n):
        selection_y.append(multi_y[i,selection[n-1-i]])
    return selection_y

def get_y_dist(path):
    loss = 0
    slope = path[-1]-path[0]
    for i in range(1,len(path)):
        loss+=abs(slope-(path[i]-path[i-1]))
    return 0

def bfs_p_appro(multi_y,X,k,edge = 0,before_loss = None):
    n=multi_y.shape[0]
    m=multi_y.shape[1]

    paths_loss = [0 for i in range(pow(m,n-1))]
    paths = np.zeros((pow(m,n-1),n))
    if edge==-1:
        for i in range(pow(m,n-1)):
            for j in range(n-1):
                index = int((i %pow(m,k-j-1))//pow(m,k-j-2))
                # print("j = ",j,"index = ",index)
                paths[i][j+1] = multi_y[j+1,index]
            min_loss = sys.maxsize
            best_sel = -1
            for p in range(m):
                paths[i][0]=multi_y[0,p]
                loss = get_y_dist(paths[i,:])
                # print(f"m={m}, p={p}, loss={loss}")
                if loss<min_loss:
                    min_loss=loss
                    best_sel = int(p*pow(m,k-2)+i//m)
            paths[i][0]=best_sel
            paths_loss[i]=min_loss

    elif edge==0:
        for i in range(pow(m,n-1)):
            for j in range(n-1):
                index = int((i %pow(m,k-j-1))//pow(m,k-j-2))
                # print("j = ",j,"index = ",index)
                paths[i][j+1] = multi_y[j+1,index]
            min_loss = sys.maxsize
            best_sel = -1
            for p in range(m):
                paths[i][0]=multi_y[0,p]
                # print("new_loss",i,p,"before_loss",p*pow(m,k-2)+i//m)
                # print("new_loss",get_k_reg(X,paths[i,:]),"before_loss",before_loss[p*pow(m,k-2)+i//m])
                loss = get_y_dist(paths[i,:]) + before_loss[p*pow(m,k-2)+i//m]
                if loss<min_loss:
                    min_loss=loss
                    best_sel = int(p*pow(m,k-2)+i//m)
            paths[i][0]=best_sel
            paths_loss[i]=min_loss

    elif edge==1:
        for i in range(pow(m,n-1)):
            for j in range(n-1):
                index = int((i %pow(m,k-j-1))//pow(m,k-j-2))
                # print("j = ",j,"index = ",index)
                paths[i][j+1] = multi_y[j+1,index]
            min_loss = sys.maxsize
            best_sel = -1
            for p in range(m):
                paths[i][0]=multi_y[0,p]
                loss = get_y_dist(paths[i,:]) + before_loss[p*pow(m,k-2)+i//m]
                if loss<min_loss:
                    min_loss=loss
                    best_sel = int(p*pow(m,k-2)+i//m)
            paths[i][0]=best_sel
            paths_loss[i]=min_loss
        # print("edge = 1: ",paths.shape)
        return paths,paths_loss
    # print("paths",paths)
    # print("paths_loss",paths_loss)
    return paths[:,0].reshape(-1,1),paths_loss

def bfs_appro(multi_y,X,k):
    n=multi_y.shape[0]
    m=multi_y.shape[1]

    sels,paths_loss = bfs_p_appro(multi_y[0:k,:],X[0:k],k,edge=-1)
    paths = sels
    for i in range(1,n-k):
        sels,paths_loss = bfs_p_appro(multi_y[i:i+k,:],X[i:i+k],k,edge=0,before_loss=paths_loss)
        paths = np.hstack((paths,sels))
    sels,paths_loss = bfs_p_appro(multi_y[n-k:n,:],X[n-k:n],k,edge=1,before_loss=paths_loss)
    paths = np.hstack((paths,sels))
    return paths,paths_loss

def appro_dp(multi_y,X,k,cluster_num=3):
    n=multi_y.shape[0]
    m=multi_y.shape[1]
    appro_graph,epsilon = get_appro_graph(n,multi_y,cluster_num)
    
    paths,paths_loss = bfs_appro(appro_graph,X,k)
    # print(paths.shape)
    # print(paths)
    best_sel = paths_loss.index(min(paths_loss))
    idx = int(paths[best_sel,n-k+1])
    cur_sel = best_sel
    path=[]
    for i in reversed(range(n-k+1)):
        next_idx = int(paths[cur_sel,i])
        path.append(multi_y[i,((next_idx)//pow(m,k-2))])
        # print("cur = ",cur_sel)
        cur_sel = next_idx
    path = list(reversed(path))
    path.extend(paths[best_sel,n-k+1:])
    return path,min(paths_loss)

def DFDP(multi_y,X,k):
    n=multi_y.shape[0]
    m=multi_y.shape[1]
    paths,paths_loss = bfs(multi_y,X,k)
    # print(paths.shape)
    # print(paths)
    best_sel = paths_loss.index(min(paths_loss))
    idx = int(paths[best_sel,n-k+1])
    cur_sel = best_sel
    selection=[]
    for i in reversed(range(n-k+1)):
        next_idx = int(paths[cur_sel,i])
        selection.append(multi_y[i,((next_idx)//pow(m,k-2))])
        cur_sel = next_idx
    selection = list(reversed(selection))
    selection.extend(paths[best_sel,n-k+1:])
    return selection,min(paths_loss)


def get_k_reg(X,sel):
    n = len(sel)
    X = np.array(X).reshape(-1,1)
    sel = np.array(sel).reshape(-1, 1)
    if (np.sum(X**2) - (1/n) * (np.sum(X))**2)==0:
        print(np.sum(X**2))
        print((1/n) * (np.sum(X))**2)
        print(X)
    w = (np.sum(X*sel)-(1/n)*np.sum(X)*np.sum(sel)) / (np.sum(X**2) - (1/n) * (np.sum(X))**2)
    b = np.mean(sel)-w*np.mean(X)
    y_pred = X*w+b
    loss = np.sum((y_pred-sel)**2)
    return loss
import copy


def bfs_p(multi_y,X,k,edge = 0,before_loss = None):
    n=multi_y.shape[0]
    m=multi_y.shape[1]

    paths_loss = [0 for i in range(pow(m,n-1))]
    paths = np.zeros((pow(m,n-1),n))
    if edge==-1:
        for i in range(pow(m,n-1)):
            for j in range(n-1):
                index = int((i %pow(m,k-j-1))//pow(m,k-j-2))
                paths[i][j+1] = multi_y[j+1,index]
            min_loss = sys.maxsize
            best_sel = -1
            for p in range(m):
                paths[i][0]=multi_y[0,p]
                loss = get_k_reg(X,paths[i,:])
                if loss<min_loss:
                    min_loss=loss
                    best_sel = int(p*pow(m,k-2)+i//m)
            paths[i][0]=best_sel
            paths_loss[i]=min_loss

    elif edge==0:
        for i in range(pow(m,n-1)):
            for j in range(n-1):
                index = int((i %pow(m,k-j-1))//pow(m,k-j-2))
                paths[i][j+1] = multi_y[j+1,index]
            min_loss = sys.maxsize
            best_sel = -1
            for p in range(m):
                paths[i][0]=multi_y[0,p]
                loss = get_k_reg(X,paths[i,:]) + before_loss[p*pow(m,k-2)+i//m]
                if loss<min_loss:
                    min_loss=loss
                    best_sel = int(p*pow(m,k-2)+i//m)
            paths[i][0]=best_sel
            paths_loss[i]=min_loss

    elif edge==1:
        for i in range(pow(m,n-1)):
            for j in range(n-1):
                index = int((i %pow(m,k-j-1))//pow(m,k-j-2))
                paths[i][j+1] = multi_y[j+1,index]
            min_loss = sys.maxsize
            best_sel = -1
            for p in range(m):
                paths[i][0]=multi_y[0,p]
                loss = get_k_reg(X,paths[i,:]) + before_loss[p*pow(m,k-2)+i//m]
                if loss<min_loss:
                    min_loss=loss
                    best_sel = int(p*pow(m,k-2)+i//m)
            paths[i][0]=best_sel
            paths_loss[i]=min_loss
        return paths,paths_loss
    # print("paths",paths)
    # print("paths_loss",paths_loss)
    return paths[:,0].reshape(-1,1),paths_loss






from sklearn.cluster import KMeans
from pyclustering.cluster.kmedians import kmedians
from sklearn.cluster import DBSCAN


def get_appro_graph(n,multi_y,k,cluster_num,method="kmean"):
    max_distance_indices = []
    m = multi_y.shape[1]
    new_graph = np.zeros((n, cluster_num))
    for i in range(n):
        if method == "kmean":
            kmeans = KMeans(n_clusters=cluster_num, random_state=42, n_init=10)
            kmeans.fit(multi_y[i].reshape(-1, 1))
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            mid = np.median(multi_y[i])
            for cluster_idx in range(cluster_num):
                cluster_points = multi_y[i][labels == cluster_idx]
                if len(cluster_points)==0:
                    new_graph[i,cluster_idx] = multi_y[i][0]
                    continue
                # print("cluster_idx:",cluster_idx,"center = ",centroids[cluster_idx],"\n",cluster_points)
                sorted_array = np.sort(cluster_points)
                if cluster_num==2 and k>3:
                    represent_idx = np.argmin(np.abs(cluster_points - mid))
                else:
                    represent_idx = np.argmin(np.abs(cluster_points - centroids[cluster_idx]))
                max_distance_indices.append(np.abs(sorted_array[-1] - sorted_array[0]))
                new_graph[i,cluster_idx] = cluster_points[represent_idx]
        elif method == "kmedian":
            initial_medians=[[0,multi_y[i,m//2-cluster_num//2+j-1]] for j in range(cluster_num)]
            kmedians_instance = kmedians(np.hstack([np.zeros((m, 1)), multi_y[i].reshape(-1, 1)]), initial_medians)
            kmedians_instance.process()
            clusters = kmedians_instance.get_clusters()
            centroids = kmedians_instance.get_medians()
            cnt = 0
            for cluster_idx in clusters:
                cluster_points = multi_y[i][ cluster_idx]
                sorted_array = np.sort(cluster_points)
                centroid = np.max(cluster_points)-np.min(cluster_points)
                represent_idx = np.argmin(np.abs(cluster_points - centroid))
                max_distance_indices.append(np.abs(sorted_array[-1] - sorted_array[0]))
                new_graph[i,cnt] = cluster_points[represent_idx]
                cnt+=1
            if cnt<m:
                for j in range(cnt,m):
                    new_graph[i:j]=new_graph[i,0]
            mid = np.median(multi_y[i])
        elif method == "DBSCAN":
            data = multi_y[i].reshape(-1, 1)
            db = DBSCAN(eps=(np.max(data)-np.min(data))/3, min_samples=1).fit(data)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            for cluster_idx in range(n_clusters_):
                cluster_points = multi_y[i][labels == cluster_idx]
                sorted_array = np.sort(cluster_points)
                centroid = np.max(cluster_points)-np.min(cluster_points)
                represent_idx = np.argmin(np.abs(cluster_points - centroid))
                max_distance_indices.append(np.abs(sorted_array[-1] - sorted_array[0]))
                new_graph[i,cluster_idx] = cluster_points[represent_idx]
            if n_clusters_<m:
                for j in range(n_clusters_,m):
                    new_graph[i:j]=new_graph[i,0]
    epsilon = max(max_distance_indices)
    return new_graph,epsilon

def DFRC(multi_y,t,k,cluster_num=2,method="kmean"):
    n=multi_y.shape[0]
    m=multi_y.shape[1]
    appro_graph,epsilon = get_appro_graph(n,multi_y,k,cluster_num,method)
    # print("epsilon",epsilon)
    selection,loss = DFDP(appro_graph,t,k)
    delta = 0
    return selection,loss,delta
    