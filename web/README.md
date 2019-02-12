## Setting Up
Install required libraries with `pip install -r requirements.txt`\
Go into web/quizkly/quizkly_app/__init__.py and change the path to the base_dir of your kvizo_core directory\
cd into quizkly and run

	python manage.py makemigrations
	python manage.py migrate
	python manage.py runserver

Common issues:
	403 error code
		Add a console.log(csrftoken) right before the fetch functions in the callbacks of form submission in src/Components/Horizontal/Interface/InnerInterface.js (sometimes token becomes inactive)