# -*- coding: utf-8 -*-
# sys.path.append("C:\Users\hedonley\Documents\projects\Leaf Area\python code")
# from traversedir import *
# from leafarea import *
# topdir = 'C:\Users\hedonley\Documents\projects\Leaf Area\leaf photos test'
# topdir = 'C:\\Users\hedonley\Documents\projects\Leaf Area\leaf photos\For Ed to Process\JB INV End ps'
# topdir = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos\\RePS Jan 15 2015\\JB INV End ps Jan 15 updated'
# traversedir(topdir,20)

def traversedir(topdir, iterationsscaled=50, itermagnification=2, debug=False, width=0, height=0, minsegmentarea=.1, minticks=10, scaleunits='mm', equalize=False, blur=False):
    "Traverse the topdir and subdirectories, looking for image files.  Combine summary data within directories."
    " Based on http://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-1/"
    "See also http://www.pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/"
    # Input:
    #   file: pathname of file to be modified
    #   iterations: number of times to apply the 3x3 structuring element in erosion and dilation operations.
    #       More iterations are needed if the petiole is wide or if blemishes are large.
    #       Note: needed iterations = 60 for petioletest02.jpg
    #       Note: needed iterations = 50 for petioletest03.jpg
    #   width: width of image in cm (or inches)
    #   height: height of image in cm (or inches)
    #   minsegmentarea: minimum area (in cm^2 or inches^2) of each leaf in image.
    #       Any smaller objects are removed.
    #   scaleunits: units for tick marks along one edge of the image.
    #       Value can be 'mm', 'cm', 'inch', 'halfinch', 'quarterinch', 'eighthinch', 'sixteenthinch'
    #       Default is 'mm'
    #       The units for the area will be cm^2 or inch^2, depending on the value of scaleunits
    # Output:
    #   newlabeled: image with petioles and small background blemishes removed and segmented into individual
    #       leaves.  Each pixel is numbered according to the leaf in which it resides.  background = 0.
    #   leafarea: an array containing the areas of leaf 1 through numleaves.
    #   numleaves: the number of leaves in the image.

    import os
    import csv
    from leafarea import *
    
    # Filename extensions for image files
    extens = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.pdf', '.eps', '.ps', '.raw']
    separator = ','
    if scaleunits == 'mm' or scaleunits == 'cm':  # scale units for ruler along edge of photo
        lengthunits = 'cm'  # units used for reporting leaf area
    else:
        lengthunits = 'in'  # units used for reporting leaf area

    numimages = 0  # keep track of the number of images processed, so that progress can be reported to the user

    # Read the old log file to determine which image files have already been processed.
    # The value of the last field, "status", is "completed" if that image file does not need to be 
    # processed.  This serves two purposes.  First, if the calculation is interrupted, the program can be
    # rerun without having to recalculate the previous results.  Second, if some of the image files
    # need additional manual preprocessing, for example using Photoshop, the user can delete the value in
    # the "status" field for those images and rerun the program.  Only the flagged images will be reprocessed.
    status = {}
    pastleaf = {}
    pastnumleaves = {}
    pastarea = {}
    pastscale = {}
    print 'Processing image files in ', os.path.realpath(topdir)
    logfilename = os.path.join(os.path.realpath(topdir), os.path.basename(os.path.normpath(topdir)) + '.csv')
    if os.path.exists(logfilename):
        with open(logfilename) as csvfile:
            reader = csv.DictReader(csvfile)
            for pastleaf in reader:
                pastdirectory = os.path.normpath(pastleaf['directory'])
                if not status.has_key((pastdirectory, pastleaf['filename'])):
                    status[(pastdirectory, pastleaf['filename'])] = ''
                if pastleaf['status'] == 'completed':
                    if pastnumleaves.has_key((pastdirectory, pastleaf['filename'])):
                        pastnumleaves[(pastdirectory, pastleaf['filename'])] += 1
                    else:
                        pastnumleaves[(pastdirectory, pastleaf['filename'])] = 1
                    pastarea[(pastdirectory, pastleaf['filename'],str(pastleaf['leaf number']))] = pastleaf['leaf area remaining (' + lengthunits + '^2)']
                    pastfill[(pastdirectory, pastleaf['filename'],str(pastleaf['leaf number']))] = pastleaf['leaf area missing (' + lengthunits + '^2)']
                    pastpercent[(pastdirectory, pastleaf['filename'],str(pastleaf['leaf number']))] = pastleaf['leaf percent missing']
                    pastscale[(pastdirectory, pastleaf['filename'])] = pastleaf['scale (' + lengthunits + '/pixel)']
                else:
                    status[(pastleaf['directory'], pastleaf['filename'])] = 'incomplete' 

    logfile = open(logfilename, 'w')
    logfile.write('directory,filename,leaf number,leaf area remaining (' + lengthunits + '^2),leaf area missing (' + lengthunits + '^2),leaf percent missing,scale (' + lengthunits + '/pixel),status\n')

    # Walk the directory tree
    for dirpath, dirnames, files in os.walk(topdir, topdown=False):

        reldirpath = os.path.normpath(os.path.relpath(dirpath, topdir))
        cleanreldirpath = reldirpath.replace(',', ';')
        # Loop through the file names for the current directory
        for file in files:
            # Split the name by '.' & get the last element
#            ext = file.lower().rsplit('.', 1)[-1]
#            print file + ext
            pathname, filename = os.path.split(file)
            basename, ext = os.path.splitext(filename)
            cleanfile = file.replace(',', ';')
            
            # If it is an image file and it is not the output of previous runs of this program, analyze the image.
            if (ext.lower() in extens) and (basename[-9:] !=  '-feedback' and basename[-9:] !=  '-leafmask'):
                if (not status.has_key((cleanreldirpath,cleanfile))) or status[(cleanreldirpath,cleanfile)] == 'incomplete': # file needs to be analyzed
    #                feedbackimagefilename = dirpath + os.sep +  basename + '-feedback' + ext
                    newimage, area, fillarea, numleaves, scale = leafarea(os.path.join(dirpath, file), iterationsscaled, itermagnification, debug, width, height, minsegmentarea, minticks, scaleunits, equalize, blur)
                    if scale == 0:
                        logfile.write(separator.join( (cleanreldirpath, cleanfile, '0', '0', '0', '0', 'no scale found along edge') ) + '\n')
                    else:
                        for leafnumber in range(numleaves):
                            logfile.write(separator.join( (cleanreldirpath, cleanfile, str(leafnumber+1), str(area[leafnumber]), str(fillarea[leafnumber]), str(fillarea[leafnumber]/(area[leafnumber]+fillarea[leafnumber])*100.0), str(scale), 'completed') ) + '\n')
                    numimages += 1
                    print numimages, ' images processed: ', file
                else:
                    # This image was successfully analyzed before.  Keep the previous results.
                    for leafnumber in range(pastnumleaves[(cleanreldirpath,cleanfile)]):
                        logfile.write(separator.join( (cleanreldirpath, cleanfile, str(leafnumber+1), str(pastarea[(cleanreldirpath,cleanfile,str(leafnumber+1))]), str(pastfill[(cleanreldirpath,cleanfile,str(leafnumber+1))]), str(pastpercent[(cleanreldirpath,cleanfile,str(leafnumber+1))]), str(pastscale[(cleanreldirpath,cleanfile)]), 'completed') ) + '\n')
                                
        # Loop through the subdirectories, combining the data for all images and subsubdirectories in that subdirectory.
#        for dir in dirnames:
            ###### I haven't implemented this.
#            print os.path.join(dirpath, dir)
    
    logfile.close()
  
    return

