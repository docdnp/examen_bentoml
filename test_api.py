"""
Pytest test suite for the Admission Predictor API.
Tests login and prediction endpoints with various scenarios.

Run with: pytest test_api.py -v
"""

import pytest
import requests


class BaseApi:
    BASE_URL = "http://localhost:3000"

    def make_credentials(self, username, password):
        """Helper to create credentials dict."""
        return {
            "credentials": {
                "username": username,
                "password": password,
            }
        }

    @pytest.fixture(scope="class")
    def valid_credentials(self):
        """Fixture to provide valid credentials for tests."""
        return self.make_credentials("user123", "password123")

    @pytest.fixture(scope="class")
    def invalid_credentials(self):
        """Fixture to provide invalid credentials for tests."""
        return self.make_credentials("invalid_user", "invalid_password")

    @pytest.fixture(scope="class")
    def valid_token(self, valid_credentials):
        """Fixture to get a valid JWT token for authenticated tests."""
        login_data = valid_credentials

        response = requests.post(f"{self.BASE_URL}/login", json=login_data)
        assert response.status_code == 200, f"Login failed: {response.text}"

        token_data = response.json()
        assert "token" in token_data, "No token in response"

        return token_data["token"]

    @pytest.fixture(scope="class")
    def invalid_token(self, invalid_credentials):
        """Fixture to get an invalid JWT token for tests."""
        login_data = invalid_credentials

        response = requests.post(f"{self.BASE_URL}/login", json=login_data)
        assert response.status_code == 401, f"Login should have failed: {response.text}"

        return "invalid_token"



class TestApiLogin(BaseApi):
    """Test class for Admission Predictor API endpoints."""

    def test_login_succeeds_with_valid_credentials(self, valid_credentials):
        """Test successful login with valid credentials."""
        login_data = valid_credentials

        response = requests.post(f"{self.BASE_URL}/login", json=login_data)

        assert response.status_code == 200
        token_data = response.json()
        assert "token" in token_data
        assert isinstance(token_data["token"], str)
        assert len(token_data["token"]) > 0

    def test_login_fails_with_invalid_credentials(self, invalid_credentials):
        """Test failed login with wrong username."""
        login_data = invalid_credentials

        response = requests.post(f"{self.BASE_URL}/login", json=login_data)

        assert response.status_code == 401
        error_data = response.json()
        assert "detail" in error_data

    def test_login_fails_with_missing_credentials(self):
        """Test failed login with missing credentials."""
        login_data = {}

        response = requests.post(f"{self.BASE_URL}/login", json=login_data)
        error_data = response.json()

        assert response.status_code == 400
        assert "detail" in error_data
        assert len(error_data["detail"]) == 1
        assert "type" in error_data["detail"][0]
        assert "loc" in error_data["detail"][0]
        assert error_data["detail"][0]["type"] == "missing"
        assert error_data["detail"][0]["loc"] == ["credentials"]


class BasePredictApi(BaseApi):
    def make_prediction_data(self, gre_score, toefl_score, cgpa):
        """Helper to create prediction input data."""
        return {
            "input_data": {
                "gre_score": gre_score,
                "toefl_score": toefl_score,
                "cgpa": cgpa
            }
        }


class TestApiPredictAccess(BasePredictApi):

    def test_prediction_fails_without_token(self):
        """Test prediction endpoint without JWT token."""
        prediction_data = self.make_prediction_data(320.0, 110.0, 8.5)

        response = requests.post(f"{self.BASE_URL}/predict", json=prediction_data)

        assert response.status_code == 401
        error_data = response.json()
        assert "detail" in error_data
        assert "authentication" in error_data["detail"].lower()

    def test_prediction_fails_with_invalid_token(self):
        """Test prediction with invalid JWT token."""
        prediction_data = self.make_prediction_data(320.0, 110.0, 8.5)

        headers = {"Authorization": "Bearer invalid_token_here"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        assert response.status_code == 401

    def test_prediction_succeeds_with_valid_token(self, valid_token):
        """Test prediction endpoint with valid JWT token."""
        prediction_data = self.make_prediction_data(320.0, 110.0, 8.5)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        assert response.status_code == 200
        result = response.json()

        assert "prediction" in result
        assert "chance_of_admission" in result
        assert "input_features" in result

        assert isinstance(result["prediction"], float)
        assert 0.0 <= result["prediction"] <= 1.0

class TestApiPredictResults(BasePredictApi):
    def test_high_chance_for_high_achiever(self, valid_token):
        """Test prediction for high-achieving student."""
        prediction_data = self.make_prediction_data(330.0, 115.0, 9.5)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        assert response.status_code == 200
        result = response.json()

        assert "prediction" in result
        assert "chance_of_admission" in result
        assert "input_features" in result

        assert isinstance(result["prediction"], float)
        assert 0.0 <= result["prediction"] <= 1.0
        assert result["prediction"] > 0.7  # High achiever should have high chance

    def test_low_chance_for_low_achiever(self, valid_token):
        """Test prediction for student with lower scores."""
        prediction_data = self.make_prediction_data(290.0, 95.0, 7.0)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        assert response.status_code == 200
        result = response.json()

        assert "prediction" in result
        assert isinstance(result["prediction"], float)
        assert 0.0 <= result["prediction"] <= 1.0
        assert result["prediction"] < 0.7  # Lower scores should have lower chance

class TestApiPredictInputValidation(BasePredictApi):
    def assert_validation_error(self, response, field, message):
        """Helper to assert validation error in response."""
        error_data = response.json()
        assert response.status_code == 400
        assert "detail" in error_data
        assert field in error_data["detail"][0]["loc"]
        assert message in error_data["detail"][0]["msg"]

    def test_fail_if_gre_too_high(self, valid_token):
        """Test input validation: GRE score too high."""
        prediction_data = self.make_prediction_data(400.0, 100.0, 8.0)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        self.assert_validation_error(response, "gre_score", "GRE Score must be between 260 and 340")


    def test_fail_if_gre_too_low(self, valid_token):
        """Test input validation: GRE score too low."""
        prediction_data = self.make_prediction_data(200.0, 100.0, 8.0)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        self.assert_validation_error(response, "gre_score", "GRE Score must be between 260 and 340")

    def test_fail_if_toefl_too_high(self, valid_token):
        """Test input validation: TOEFL score too high."""
        prediction_data = self.make_prediction_data(320.0, 150.0, 8.0)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        self.assert_validation_error(response, "toefl_score", "TOEFL Score must be between 0 and 120")

    def test_fail_if_toefl_too_low(self, valid_token):
        """Test input validation: TOEFL score negative."""
        prediction_data = self.make_prediction_data(320.0, -10.0, 8.0)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        self.assert_validation_error(response, "toefl_score", "TOEFL Score must be between 0 and 120")

    def test_fail_if_cgpa_too_high(self, valid_token):
        """Test input validation: CGPA too high."""
        prediction_data = self.make_prediction_data(320.0, 100.0, 15.0)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        # CGPA must be between 0 and 10
        self.assert_validation_error(response, "cgpa", "CGPA must be between 0 and 10")

    def test_fail_if_cgpa_too_low(self, valid_token):
        """Test input validation: CGPA negative."""
        prediction_data = self.make_prediction_data(320.0, 100.0, -1.0)

        headers = {"Authorization": f"Bearer {valid_token}"}

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data,
            headers=headers
        )

        self.assert_validation_error(response, "cgpa", "CGPA must be between 0 and 10")

