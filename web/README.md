## Setting Up Backend
Install required libraries with `pip install -r requirements.txt`\
Migrate models with command
'python manage.py makemigrations'
`python manage.py migrate`
Alter or create a config.json file that specifies absolute paths to the models
An examples is be as follows:
`{
	"sentence_model": "../../local/sentencemodel/1550462322",
	"gap_model": "../../local/gap_model",
	"word_model": "../../local/wmdatabio70.bin"
}`
Then start server.
`python web/quizkly/python manage.py runserver --config_file="[path] to config.json"`

## Setting Up Frontend

### Install Vue CLI
`npm install -g @vue/cli`

### Project setup
```
npm install
```

#### Compiles and hot-reloads for development
```
npm run serve
```

#### Compiles and minifies for production
```
npm run build
```

#### Run your tests
```
npm run test
```

#### Lints and fixes files
```
npm run lint
```

#### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).
