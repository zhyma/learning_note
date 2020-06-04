'''
Based on https://www.polarxiong.com/archives/Python-%E7%94%9F%E6%88%90%E7%9B%AE%E5%BD%95%E6%A0%91.html
This script do two things:
1. Check all folders, create a HTML menu as list.
2. Chekc all blog files, replace "*.md" with "*.html"
'''
import os
import os.path

BRANCH = '├─'
LAST_BRANCH = '└─'
TAB = '│  '
EMPTY_TAB = '   '

def process_html(filename):
    f = open(filename, "r", encoding='utf8')
    content = f.read()
    print(filename)

def get_dir_list(path, placeholder='', pastpass=''):
    folder_list = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    file_list = [file for file in os.listdir(path) if (os.path.isfile(os.path.join(path, file)) and file.split('.')[-1]=='html')]
    result = ''
    ph = placeholder

    for idx, folder in enumerate(folder_list):
        t = TAB
        b = BRANCH
        if not idx == len(folder_list)-1:
            if not file_list:
                t = EMPTY_TAB
                b = LAST_BRANCH

        sublist = get_dir_list(os.path.join(path, folder), placeholder + t, pastpass + folder + '/')
        # skip empty folder
        if 'html' in sublist:
            result += placeholder + b + '<b>' + folder + '</b>' + '<br />' + sublist


    for idx, file in enumerate(file_list):
        if idx == len(file_list)-1:
            b = LAST_BRANCH
        else:
            b = BRANCH
        process_html(pastpass+file)
        result += ph + b + '<a href="' + pastpass + file + '">' + file + '</a><br />'

    result += ph+'<br />'

    return result


if __name__ == '__main__':
    htmlfile = get_dir_list(os.getcwd(),'')
    f = open('index.html', 'w', encoding='utf8')
    htmlfile = '<!doctype html>\n<html>\n<head></head><body>' + htmlfile + '</body>'
    f.write(htmlfile)
    f.close()