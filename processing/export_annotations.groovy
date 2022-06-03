/* 
 * Prerequisite: have a ".qpdata" file open in QuPath.
 * 
 * Instructions:
 * 
 * On a Mac:
 * - To open this script in QuPath: Automate -> Show Script editor.
 * - To run this script in QuPath: Command-R.
 */

// Replace this file path with the location to save the exported annotations.
// Note that if you have a read-only file system, you may need to create an 
// empty file called 'annotations.csv' before running this script.
def file_path = 'annotations.csv'

def result = 'Name,X,Y,Width,Height'
result += System.lineSeparator()
for (annotation in getAnnotationObjects()) {
    def name = annotation.getName()
    def roi = annotation.getROI()
    result += String.format('%s,%.2f,%.2f,%.2f,%.2f',
        name, roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight())
    result += System.lineSeparator()
}

print result
def file = new File(file_path)
file.text = result
