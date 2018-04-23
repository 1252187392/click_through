#encoding:utf-8
import sys
import os
'''
将filename进行切分，每nums行划分为一个文件
'''
assert len(sys.argv) > 2
filename = sys.argv[1]
nums = int(sys.argv[2])

print 'split {} with nums {}'.format(filename,nums)
cnt = 0
with open(filename) as fin:
    dir_name = filename.replace('.csv','')
    os.system('mkdir ' + dir_name)
    outfile = filename.replace('.csv','') + '/0.csv'
    fout = open(outfile,'w')
    header = next(fin).strip()
    print >> fout,header
    for line in fin:
        print >> fout,line.strip()
        cnt += 1
        if cnt % nums == 0:
            fout.close()
            outfile = filename.replace('.csv', '') + '/{}.csv'.format(cnt/nums)
            fout = open(outfile, 'w')
            print >>fout,header
    fout.close()

print 'split {} over'.format(filename)