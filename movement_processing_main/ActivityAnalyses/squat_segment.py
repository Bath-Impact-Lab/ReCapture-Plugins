# %% Segment squats.
def segment_squats(ikFilePath, pelvis_ty=None, timeVec=None, visualize=False,
                  filter_pelvis_ty=True, cutoff_frequency=4, height=.2):
    
    # TODO: eventually, this belongs in a squat_analysis class and should take
    # the form of segment_gait
    
    # Extract pelvis_ty if not given.
    if pelvis_ty is None and timeVec is None:
        ikResults = storage_to_dataframe(ikFilePath,headers={'pelvis_ty'})
        timeVec = ikResults['time']
        if filter_pelvis_ty:
            from utilsOpenSimAD import filterNumpyArray
            pelvis_ty = filterNumpyArray(
                ikResults['pelvis_ty'].to_numpy(), timeVec.to_numpy(), 
                cutoff_frequency=cutoff_frequency)
        else:
            pelvis_ty = ikResults['pelvis_ty']    
    dt = timeVec[1] - timeVec[0]

    # Identify minimums.
    pelvSignal = np.array(-pelvis_ty - np.min(-pelvis_ty))
    pelvSignalPos = np.array(pelvis_ty - np.min(pelvis_ty))
    idxMinPelvTy,_ = signal.find_peaks(pelvSignal,distance=.7/dt,height=height)
    
    # Find the max adjacent to all of the minimums.
    minIdxOld = 0
    startFinishInds = []
    for i, minIdx in enumerate(idxMinPelvTy):
        if i<len(idxMinPelvTy)-1:
            nextIdx = idxMinPelvTy[i+1]
        else:
            nextIdx = len(pelvSignalPos)
        startIdx = np.argmax(pelvSignalPos[minIdxOld:minIdx]) + minIdxOld
        endIdx = np.argmax(pelvSignalPos[minIdx:nextIdx]) + minIdx
        startFinishInds.append([startIdx,endIdx])
        minIdxOld = np.copy(minIdx)
    startFinishTimes = [timeVec[i].tolist() for i in startFinishInds]
    
    if visualize:
        plt.figure()     
        plt.plot(-pelvSignal)
        for c_v, val in enumerate(startFinishInds):
            plt.plot(val, -pelvSignal[val], marker='o', markerfacecolor='k',
                     markeredgecolor='none', linestyle='none',
                     label='Squatting phase')
            if c_v == 0:
                plt.legend()
        plt.xlabel('Frames')
        plt.ylabel('Position [m]')
        plt.title('Vertical pelvis position')
        plt.draw()
    
    return startFinishTimes