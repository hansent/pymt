from distutils.core import setup
from distutils.sysconfig import get_python_lib
import sys


sys.argv.append('install')
pylib = get_python_lib()
setup (name='pymt',
      version='0.1',
      author='Thomas Hansen',
      author_email='thomas.hansen@gmail.com',
      url='http://code.google.com/p/pymt/',
      license='GPL',
      packages=['pymt', 'pymt.lib','pymt.lib.osc', 'pymt.ui', 'pymt.ui.widgets',
               'pymt.ui.widgets.layout', 'pymt.ui.widgets.form'],
      package_dir={'pymt': 'pymt'},
      package_data={'pymt': ['data/icons/svg/*.svg', 'data/icons/*.png', 'data/colors.json']}
)
