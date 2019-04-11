python -m unittest discover -p '*_test.py'
RET1=$?
flake8
RET2=$?
cd web/quizkly
python manage.py test
RET3=$?
if [ $RET1 -ne 0 ] || [ $RET2 -ne 0 ] || [ $RET3 -ne 0 ]
then
	exit 1
else
	exit 0
fi
