python -m unittest discover -p '*_test.py'
RET1=$?
flake8
RET2=$?
cd web/quizkly

rm -r quizkly_app/migrations
python manage.py makemigrations quizkly_app
python manage.py migrate
python manage.py test --noinput
cd ../..
RET3=$?
cd ../frontend
if [ $RET1 -ne 0 ] || [ $RET2 -ne 0 ] || [ $RET3 -ne 0 ]
then
	exit 1
else
	exit 0
fi
