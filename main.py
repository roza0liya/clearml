from clearml import Task
import functools
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(input_file):
    return pd.read_csv(input_file)

def clearml_task(project_name, task_name=None, tags=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            automatic_task_name = task_name or func.__name__
            task = Task.init(
                project_name=project_name,
                task_name=automatic_task_name,
                tags=tags or []
            )
            
            # Подключение аргументов к задаче
            task.connect(args)
            task.connect(kwargs)

            try:
                result = func(*args, **kwargs)
                task.get_logger().report_scalar(
                    title='Execution',
                    series='Success',
                    value=1,
                    iteration=0
                )

                # Загрузка артефактов
                task.upload_artifact(name='processed_data', artifact_object=result)
                
                # Пример логирования графика
                fig, ax = plt.subplots()
                ax.plot(np.random.rand(10))
                task.get_logger().report_matplotlib_figure(
                    title='Random Plot', 
                    series='Test Series', 
                    figure=fig
                )

                return result

            except Exception as e:
                task.get_logger().report_scalar(
                    title='Execution',
                    series='Error',
                    value=1,
                    iteration=0
                )
                task.get_logger().report_text(str(e))
                raise

        return wrapper
    return decorator

@clearml_task(
    project_name="ML_Lab_Experiments",
    task_name="Data_Processing",
    tags=["preprocessing", "v1"]
)
def process_data(input_file, threshold=0.5):
    # Ваш код обработки данных
    data = load_data(input_file)
    processed_data = data[data['score'] > threshold]
    return processed_data

# Использование декоратора
result = process_data(r'C:\Users\rozal\Desktop\МАГ 1 курс\Питон\dataset.csv', threshold=0.7)