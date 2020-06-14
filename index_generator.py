'''
Based on https://www.polarxiong.com/archives/Python-%E7%94%9F%E6%88%90%E7%9B%AE%E5%BD%95%E6%A0%91.html
This script do two things:
1. Check all folders, create a HTML menu as list.
2. Chekc all blog files, replace "*.md" with "*.html". (TODO)
'''
import os
import os.path

BRANCH = '├─'
LAST_BRANCH = '└─'
TAB = '│&nbsp;&nbsp;'
EMPTY_TAB = '&nbsp;&nbsp;&nbsp;'

def process_html(filename):
    f = open(filename, "r", encoding='utf8')
    content = f.read()
    print(filename)

def get_dir_list(path, placeholder='', pastpass=''):
    folder_list = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    file_list = [file for file in os.listdir(path) if (os.path.isfile(os.path.join(path, file)) and file.split('.')[-1]=='html')]
    result = ''
    ph = placeholder

    # result += ph + TAB + '<br />'

    for idx, file in enumerate(file_list):
        if idx == len(file_list)-1 and len(folder_list) == 0:
            b = LAST_BRANCH
        else:
            b = BRANCH
        process_html(pastpass+file)
        result += ph + b + '<a href="' + pastpass + file + '">' + file + '</a><br />'

    if len(folder_list) == 0:
        result += ph + '<br />'
    else:
        result += ph + TAB + '<br />'

    for idx, folder in enumerate(folder_list):
        t = TAB + EMPTY_TAB
        b = BRANCH
        if idx == len(folder_list)-1:
            t = EMPTY_TAB + EMPTY_TAB
            b = LAST_BRANCH

        sublist = get_dir_list(os.path.join(path, folder), ph + t, pastpass + folder + '/')
        # skip empty folder
        if 'html' in sublist:
            result += ph + b + '<b>' + folder + '</b>' + '<br />' + sublist

    return result


if __name__ == '__main__':
    htmlfile = get_dir_list(os.getcwd())
    f = open('index.html', 'w', encoding='utf8')
    htmlfile = '<!doctype html>\n<html>\n<head></head><body>' + htmlfile + '</body>'
    f.write(htmlfile)
    f.close()