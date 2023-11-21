import React, { useCallback, useContext, useState } from "react";
import {
  createAnimation,
  IonModal,
  IonButton,
  IonContent,
  IonInput,
} from "@ionic/react";
import { AuthContext } from "../auth/AuthProvider";

interface SignupState {
  username?: string;
  password?: string;
}

export const MyModal: React.FC = () => {
  const [showModal, setShowModal] = useState(false);

  const enterAnimation = (baseEl: any) => {
    const root = baseEl.shadowRoot;
    const backdropAnimation = createAnimation()
      .addElement(root.querySelector("ion-backdrop")!)
      .fromTo("opacity", "0.01", "var(--backdrop-opacity)");

    const wrapperAnimation = createAnimation()
      .addElement(root.querySelector(".modal-wrapper")!)
      .keyframes([
        { offset: 0, opacity: "0", transform: "scale(0)" },
        { offset: 1, opacity: "0.99", transform: "scale(1)" },
      ]);

    return createAnimation()
      .addElement(baseEl)
      .easing("ease-out")
      .duration(500)
      .addAnimation([backdropAnimation, wrapperAnimation]);
  };

  const leaveAnimation = (baseEl: any) => {
    return enterAnimation(baseEl).direction("reverse");
  };

  const [state, setState] = useState<SignupState>({});
  const { username, password } = state;
  const { signup } = useContext(AuthContext);
  const handlePasswwordChange = (e: any) =>
    setState({
      ...state,
      password: e.detail.value || "",
    });
  const handleUsernameChange = (e: any) =>
    setState({
      ...state,
      username: e.detail.value || "",
    });
  const handleSignUp = useCallback(() => {
    console.log("handleSignup...");
    signup?.(username, password);
  }, [username, password]);

  return (
    <>
      <IonModal
        isOpen={showModal}
        enterAnimation={enterAnimation}
        leaveAnimation={leaveAnimation}
      >
        <IonInput
          placeholder="Username"
          value={username}
          onIonChange={handleUsernameChange}
        />
        <IonInput
          placeholder="Password"
          value={password}
          onIonChange={handlePasswwordChange}
        />
        <IonButton
          onClick={() => {
            setShowModal(false);
            handleSignUp();
          }}
        >
          Register
        </IonButton>
      </IonModal>
      <IonButton onClick={() => setShowModal(true)}>Sign Up</IonButton>
    </>
  );
};
