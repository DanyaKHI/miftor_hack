## Запуск проекта
Нужно добавить веса ruBert512 для этого возьмите файлик [model.safetensors](https://disk.yandex.ru/d/Wl1NVdl2mkVx9g) -> models/ruBert512


### В docker container
```
docker-compose up
```
### Инференс файлика из директории

Расскоментируйте строчки

```
# TODO Расскомментировать для обычного инференса
# df = pd.read_csv('gt_test.csv')
# df_result = predict(df)
# output_file = 'submission.csv'
```