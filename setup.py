from setuptools import setup, find_packages

setup(name='cancer_model', #nasam
      version='0.1',
      description='software management',
      author='Faizrakhmanov I.R.', #Samikaev N.R.
      author_email='ilnur-qwerty@mail.ru', #samikaevn@yandex.ru
      packages=find_packages(),
      install_requires=[
          'dash',
          'dash_core_components',
          'dash_html_components',
          'torch',
          'sklearn'
      ],
      package_data={'cancer_model': ['model_state_dict.pt', 'mms.pickle']}, #package_data={'nasam': ('ready model', 'cancer_train.csv')}
      include_package_data=True,
      zip_safe=False)
