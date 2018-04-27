'''
Input: XML files and pickle files

Output:

X | Y

X:
Current shot and previous shot info:
-Diff between shot frames (use avg for last shot)
x-Angle b/w shot and center line
-Player and opponent court position when shot starts
-Shot start location
-Shot end location (use avg for last shot)

-Type of shot
-Player and opponent court position when shot ends

Y (Shot outcome):
1 - Server wins
0 - Receiver wins
'''

import cPickle as pickle
import os
from multiprocessing.dummy import Pool as ThreadPool
import xml.etree.cElementTree as ET
# import subprocess as sp
import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from pykalman import KalmanFilter

pkl_base_pth = '/users/rakesh.jasti/badminton/BadmintonDatasetFull/event_det/pkl'
xml_in_p = '/users/rakesh.jasti/badminton/Homo/xml/edit'
event_feat_pkl_p = '/users/rakesh.jasti/badminton/BadmintonDatasetFull/event_feat'

baseline_pt = 134
baseline_pb = 1407


def apply_kalman_physics_model(coordinates, vis_name='test', vis=False):
    initial_state_mean = [coordinates[0, 0],0,
                          coordinates[0, 1],0]
    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]
    observation_matrix = [[1, 0, 0, 0],
                         [0, 0, 1, 0]]
    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)

    kf1 = kf1.em(coordinates, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(coordinates)

    # smoothen out further
    kf2 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean,
                  observation_covariance = 10*kf1.observation_covariance,
                  em_vars=['transition_covariance', 'initial_state_covariance'])
    kf2 = kf2.em(coordinates, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(coordinates)


    # speed = np.sqrt(np.square(smoothed_state_means[:, 1]) + np.square(smoothed_state_means[:, 3]))#*fps*scale*3.6 # in km/h
    speed_x = smoothed_state_means[:, 1]
    # print speed.shape
    # speed = speed[(speed[:]>0) & (speed[:]<max_speed)];
    avg_speed_x = float(np.average(speed_x))#*fps*scale*3.6 # in km/h
    dist_x = avg_speed_x * len(coordinates)

    # # plot fiter by kalman
    if vis:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        times = range(coordinates.shape[0])
        ax.plot(times, coordinates[:, 0], 'bo',
            times, coordinates[:, 1], 'ro',
            times, smoothed_state_means[:, 0], 'b--',
            times, smoothed_state_means[:, 2], 'r--',)
        fig.savefig('vis_outputs/'+vis_name+'.png')
        plt.close(fig)
    
    # if avg_speed > max_avg_speed:
    #     print vis_name,
    #     print 'speed: {}'.format(avg_speed)
    #     return [None]*2
    return dist_x


def lateral_movement_fn(rally, frame_idx, frame_idx_1):
    top_pos = np.empty((0,2), int);
    bot_pos = np.empty((0,2), int);

    for i in range(frame_idx, frame_idx_1+1):
        top_pos=np.append(top_pos, np.array([[ int(rally[i][0].get('x')), int(rally[i][0].get('y')) ]]), axis=0)
        bot_pos=np.append(bot_pos, np.array([[ int(rally[i][1].get('x')), int(rally[i][1].get('y')) ]]), axis=0)

    lateral_pt = apply_kalman_physics_model(top_pos, vis_name=str(frame_idx)+'_pt', vis=False)
    lateral_pb = apply_kalman_physics_model(bot_pos, vis_name=str(frame_idx)+'_pb', vis=False)
    return abs(lateral_pt / float(lateral_pb))

def angle(x1,y1, x2, y2):
    if x1==x2:
        return 90
    m = (y2-y1) / float(x2-x1) 
    
    return math.atan(m) *180/math.pi

def processRally(rally, match_name, k, num_of_feats):

    label = 1 if rally.get('winner') == rally.get('service') else 0

    rally_id=rally.get('num')
    print 'rally id: '+str(rally_id)
    pkl_pth=os.path.join(pkl_base_pth, match_name, 'rally_{}.pkl'.format(rally_id))
    event_preds=pickle.load(open(pkl_pth)) #event preds are zero indexed

    # remove no event, react events & remove short events
    event_preds=event_preds[ np.where((event_preds[:,3]!=1) * (event_preds[:,3]!=2) * (event_preds[:,2]>5))]

    if len(event_preds[:,0]) <= 2:
        return None

    # # get avg react frame diff (to calculate last event feats)
    # add third column for mid_point of the event
    mid_frame = (event_preds[:,0]+event_preds[:,1])/2+1
    avg_diff = sum(mid_frame[1:] - mid_frame[:-1]) / len(mid_frame[1:])


    pkl_p = os.path.join(event_feat_pkl_p, match_name, 'rally_'+str(rally_id)+'.pkl')
    if os.path.isfile(pkl_p):
        print "Loading "+'rally_{}.pkl'.format(rally_id)
        with open(pkl_p, 'rb') as pkl:
            event_feats = pickle.load(pkl)
    else:
        event_feats = np.empty((0,num_of_feats), float);
        for i, event in enumerate(event_preds):

            frame_idx = (event[0]+event[1])/2+1
            if len(event_preds) == i+1:
                frame_idx_1 = frame_idx+avg_diff
                frame_idx_1 = min(frame_idx_1, len(rally)-1)
            else:
                frame_idx_1 = (event_preds[i+1][0]+event_preds[i+1][1])/2+1
            
            # shot speed (frame id diff)
            shot_speed = frame_idx_1 - frame_idx
            
            # shot end loc, Baseline distance ratio, Lateral player movement ratio
            if event[3] in [3,5,7,9,11]:    #pt
                shot_start_loc = [ int(rally[frame_idx][0].get('x')), int(rally[frame_idx][0].get('y')) ]
                shot_end_loc = [ int(rally[frame_idx_1][1].get('x')), int(rally[frame_idx_1][1].get('y')) ]
                try:
                    baseline_ratio = ( int(rally[frame_idx_1][0].get('y')) - baseline_pt ) / ( baseline_pb - int(rally[frame_idx_1][1].get('y')) )
                    lateral_movement_ratio = lateral_movement_fn(rally, frame_idx, frame_idx_1)
                except ZeroDivisionError:
                    return None

                shot_theta = abs( angle(shot_start_loc[0], shot_start_loc[1], shot_end_loc[0], shot_end_loc[1]) )

            else:   #pb
                shot_start_loc = [ int(rally[frame_idx][1].get('x')), int(rally[frame_idx][1].get('y')) ]
                shot_end_loc = [ int(rally[frame_idx_1][0].get('x')), int(rally[frame_idx_1][0].get('y')) ]

                try:
                    baseline_ratio = ( ( baseline_pb - int(rally[frame_idx_1][1].get('y')) / int(rally[frame_idx_1][0].get('y')) - baseline_pt ) )
                    lateral_movement_ratio = 1. / lateral_movement_fn(rally, frame_idx, frame_idx_1)
                except ZeroDivisionError:
                    return None

                shot_theta = abs( angle(shot_start_loc[0], shot_start_loc[1], shot_end_loc[0], shot_end_loc[1]) )


            # include shot start location, angle with the center line
            # pt, pb, shot_speed, shot_end_loc, shot_type, baseline_ratio, lateral_movement_ratio
            shot_feats = np.array([[ int(rally[frame_idx][0].get('x')), int(rally[frame_idx][0].get('y')), \
                int(rally[frame_idx][1].get('x')), int(rally[frame_idx][1].get('y')), \
                shot_speed, \
                # shot_start_loc[0], shot_start_loc[1], \
                shot_end_loc[0], shot_end_loc[1], \
                shot_theta, \
                event[3], \
                baseline_ratio, \
                lateral_movement_ratio \
                ]])
            event_feats = np.append(event_feats, shot_feats, axis=0)

        # save rally feats in pkl
        with open(pkl_p, 'wb') as pkl:
            pickle.dump(event_feats, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    # Create final data
    rally_data = np.empty((0, num_of_feats*(k)), float)
    for i, feats in enumerate(event_feats[k-1:,:], k-1):
        shot_data = event_feats[i].tolist()
        
        for j in range(1, k):
            shot_data = shot_data + event_feats[i-j, :].tolist()
        
        rally_data = np.vstack(( rally_data, np.asarray(shot_data).reshape(1,num_of_feats*(k)) ))
    
    # add Y to the data
    rally_data = np.hstack((rally_data, np.ones((rally_data.shape[0],1))*label))

    return rally_data

def processMatch(match_name, k, num_of_feats):
    print match_name

    match_xml_p = os.path.join(xml_in_p, match_name+'.xml')
    match_xml = ET.parse(match_xml_p).getroot()
    
    match_feats = np.empty((0, num_of_feats*(k)+1), float)

    for i, rally in enumerate(match_xml):
        # if i > 1:
        #     break
        rally_data = processRally(rally, match_name, k, num_of_feats)
        if rally_data is not None:
            match_feats = np.vstack(( match_feats, rally_data ))

    return match_feats

train_matches = [
'Baun-Augustyn-GrpG-LondonOlympics-2012',
'Chen-Wacha-GrpL-LondonOlympics-2012',
'Chen-Zwiebler-R32-LondonOlympics-2012',
'Cordon-Hurskainen-GrpM-LondonOlympics-2012',
'Dan-Evans-GrpP-LondonOlympics-2012',
'Firdasari-Zaitsava-GrpO-LondonOlympics-2012',
'Hidayat-Abian-GrpO-LondonOlympics-2012',
'Karunaratne-Parupalli-R16-LondonOlympics-2012',
'Lee-Chen-QtrFinals-LondonOlympics-2012',
'Lee-Dan-SemiFinals-LondonOlympics-2012',
'Lee-Long-Bronze-LondonOlympics-2012',
'Li-Wang-SemiFinals-LondonOlympics-2012',
'Magee-Hosny-GrpI-LondonOlympics-2012',
'Na-Fasungova-GrpD-LondonOlympics-2012',
'Nehwal-Xin-Bronze-LondonOlympics-2012',
'Nguyen-Parupalli-GrpD-LondonOlympics-2012',
'Nguyen-Tan-GrpD-LondonOlympics-2012',
'Sasaki-Cordon-R16-LondonOlympics-2012',
'Sasaki-Soeroredjo-GrpN-LondonOlympics-2012',
'Shenck-Gavnholt-GrpN-LondonOlympics-2012',
'Sung-Yip-GrpJ-LondonOlympics-2012',
'Wang-Li-Finals-LondonOlympics-2012',
]
val_matches = [
'WeiLee-Dan-Finals-LondonOlympics-2012',
'WeiLee-Lang-GrpA-LondonOlympics-2012',
'WeiLee-Long-SemiFinals-LondonOlympics-2012',
'WeiLee-Parupalli-QtrFinals-LondonOlympics-2012',
'Yihan-Nehwal-SemiFinals-LondonOlympics-2012',
]

if __name__ == '__main__':

    k = 2   # k is the number of timesteps to be considered; if k=2, then consider t and t-1
    num_of_feats = 11

    train_feats = np.empty((0, num_of_feats*(k)+1), float)
    val_feats = np.empty((0, num_of_feats*(k)+1), float)

    train_pkl_p = 'data/train.pkl'
    val_pkl_p = 'data/val.pkl'

    for match in train_matches:
        if not os.path.exists(os.path.join(event_feat_pkl_p, match)):
            os.makedirs(os.path.join(event_feat_pkl_p, match))

        train_feats = np.vstack ((train_feats, processMatch(match, k, num_of_feats)))

    with open(train_pkl_p, 'wb') as pkl:
        pickle.dump(train_feats, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    for match in val_matches:
        if not os.path.exists(os.path.join(event_feat_pkl_p, match)):
            os.makedirs(os.path.join(event_feat_pkl_p, match))

        val_feats = np.vstack ((val_feats, processMatch(match, k, num_of_feats)))

    with open(val_pkl_p, 'wb') as pkl:
        pickle.dump(val_feats, pkl, protocol=pickle.HIGHEST_PROTOCOL)

