integral(@sin, 0, pi)
trapez(@sin, 0, pi, 200)
Simpson(@sin, 0, pi, 200)
drept(@sin, 0, pi, 200)
disp("Quad");
quad(@sin, 0, pi)
adaptquad(@sin, 0, pi, 0.0001, @trapez)
adaptquad(@sin, 0, pi, 0.0001, @Simpson)
adaptquad(@sin, 0, pi, 0.0001, @drept)
disp("Romberg");
Romberg(@sin, 0, pi)
disp("AdQuad");
adquad(@sin, 0, pi, 0.0001)