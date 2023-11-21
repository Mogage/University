import React, { useEffect, useRef } from "react";
import { CreateAnimation, createAnimation } from "@ionic/react";
import { MyComponent } from "./MyComponent";
import "./AnimationDemo.css";
import { MyModal } from "./MyModal";

const AnimationDemo: React.FC = () => {
  const elCRef = useRef(null);
  const animationRef = useRef<CreateAnimation>(null);
  useEffect(simpleAnimationJS, []);
  useEffect(groupAnimations, []);
  useEffect(chainAnimations, []);
  useEffect(simpleAnimationReact, [animationRef.current]);
  return (
    <div className="container">
      <div className="square-a">
        <p>Simple Animation - createAnimation</p>
      </div>
      <CreateAnimation
        ref={animationRef}
        duration={5000}
        fromTo={{
          property: "transform",
          fromValue: "translateY(0) rotate(0)",
          toValue: `translateY(200px) rotate(180deg)`,
        }}
        easing="ease-out"
      >
        <div>Simple Animation - CreateAnimation</div>
      </CreateAnimation>
      <div className="square-b">
        <p>Grouped animation 1</p>
      </div>
      <div ref={elCRef}>
        <p>Grouped animation 2</p>
      </div>
      <MyComponent />
      <MyModal />
    </div>
  );

  function simpleAnimationJS() {
    const el = document.querySelector(".square-a");
    if (el) {
      const animation = createAnimation()
        .addElement(el)
        .duration(5000)
        .direction("alternate")
        .iterations(Infinity)
        .keyframes([
          { offset: 0, transform: "scale(3)", opacity: "1" },
          { offset: 0.5, transform: "scale(1.5)", opacity: "1" },
          {
            offset: 1,
            transform: "scale(0.5)",
            opacity: "0.2",
          },
        ]);
      animation.play();
    }
  }

  function simpleAnimationReact() {
    if (animationRef.current !== null) {
      animationRef.current.animation.play();
    }
  }

  function groupAnimations() {
    const elB = document.querySelector(".square-b");
    if (elB && elCRef.current) {
      const animationA = createAnimation()
        .addElement(elB)
        .fromTo("transform", "scale(1)", "scale(1.5)");
      const animationB = createAnimation()
        .addElement(elCRef.current)
        .fromTo("transform", "scale(1)", "scale(0.5)");
      const parentAnimation = createAnimation()
        .duration(5000)
        .addAnimation([animationA, animationB]);
      parentAnimation.play();
    }
  }

  function chainAnimations() {
    const elB = document.querySelector(".square-b");
    if (elB && elCRef.current) {
      const animationA = createAnimation()
        .addElement(elB)
        .duration(5000)
        .fromTo("transform", "scale(1)", "scale(1.5)")
        .afterStyles({
          background: "green",
        });
      const animationB = createAnimation()
        .addElement(elCRef.current)
        .duration(7000)
        .fromTo("transform", "scale(1)", "scale(0.5)")
        .afterStyles({
          background: "green",
        });
      (async () => {
        await animationA.play();
        await animationB.play();
      })();
    }
  }
};

export default AnimationDemo;
