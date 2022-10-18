package model;

import java.util.Objects;

public abstract class Task {
    private String taskId;
    private String description;

    public Task(String _taskId, String _description){
        this.taskId = _taskId;
        this.description = _description;
    }

    public void setTaskId(String taskId){
        this.taskId = taskId;
    }

    public String getTaskId() {
        return taskId;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }

    @Override
    public String toString() {
        return this.taskId + " " + this.description;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o){
            return true;
        }
        if(!(o instanceof Task task)){
            return false;
        }
        return Objects.equals(getTaskId(), task.getTaskId()) &&
                Objects.equals(getDescription(), task.getDescription());
    }

    @Override
    public int hashCode(){
        return Objects.hash(getTaskId(), getDescription());
    }

    public abstract void execute();
}













