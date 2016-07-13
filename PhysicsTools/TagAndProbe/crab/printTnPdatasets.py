import tnpDatasetDef as tnpDSet

campaign = {}
for ds in tnpDSet.datasetsForTnP:
    if not ds['campaign'] in campaign.keys():
        campaign[ds['campaign']] = {}
    campaign[ds['campaign']][ds['dataset']] = {'lumi': ds['lumi'],'nEvts': ds['nEvts'], 'file':ds['file'] }
    

for cp in campaign:
    print ' ******** campaign :', cp
    for dataset in campaign[cp]:
        print '   - dataset: %-*s nEvts: %-*d lumi: %1.3f (/fb)\t, file: %s' %(60, dataset,
                                                                               10, campaign[cp][dataset]['nEvts'],
                                                                                campaign[cp][dataset]['lumi'],
                                                                                campaign[cp][dataset]['file'])
                                                                 

